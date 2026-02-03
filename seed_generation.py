import json
from pathlib import Path
from typing import List

from langgraph.graph import StateGraph, START, END

from agent.seed_generation.correct import correction_agent, summarize_correction_history
from agent.seed_generation.table_selection import table_selection_agent
from agent.seed_generation.generation import generation_agent
from agent.seed_generation.critical import critical
from agent.seed_generation.ir_generation import ir_generation_agent
from agent.seed_generation.alignment import alignment
from state.seed_generation_state import SeedGenerationState
import util.postgres_util as postgres_util
import util.oracle_util as oracle_util
from tool.seed_generation_tool import database_selection, init_database_statistics, poisson_sample, random_walk_table_selection
from config.common import pg_config, oc_config, seed_generation_config

_PG_CORRECTION_LOG_PATH = Path(seed_generation_config['postgres_correction_experiences_path'])
_OC_CORRECTION_LOG_PATH = Path(seed_generation_config['oracle_correction_experiences_path'])

table_selection_redundancy = 3
postgres_db_schema_graph = postgres_util.get_database_schema_graph()
oracle_db_schema_graph = oracle_util.get_database_schema_graph()
postgres_db_schema_dict = postgres_util.get_database_schema_json()
oracle_db_schema_dict = oracle_util.get_database_schema_json()


def _get_correction_log_path(dialect: str) -> Path:
    if dialect == "postgresql":
        return _PG_CORRECTION_LOG_PATH
    if dialect == "oracle":
        return _OC_CORRECTION_LOG_PATH
    raise ValueError(f"Unsupported dialect '{dialect}'. Expected 'postgresql' or 'oracle'.")


def _append_correction_summary(summary_record: dict, dialect: str) -> None:
    if not summary_record or not isinstance(summary_record, dict):
        return
    required_keys = {"original_plsql", "corrected_plsql", "correction_experience"}
    if not required_keys.issubset(summary_record.keys()):
        return

    log_path = _get_correction_log_path(dialect)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8") as fp:
                entries = json.load(fp)
            if not isinstance(entries, list):
                entries = []
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append(summary_record)

    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(entries, fp, ensure_ascii=False, indent=2)


def entry_node(state: SeedGenerationState) -> SeedGenerationState:
    state["epoch"] = 0
    state["current_plsql_number"] = 0
    return state


def dialect_node(state: SeedGenerationState) -> SeedGenerationState:
    state["database_statistics"] = init_database_statistics(state["dialect"])
    return state


def management_node(state: SeedGenerationState) -> SeedGenerationState:
    state["epoch"] += 1
    state["critical_epoch"] = 0
    state["generated_plsqls"] = []
    return state


def route_on_whether_continue_generation(state: SeedGenerationState) -> str:
    if state["current_plsql_number"] >= state["target_plsql_number"]:
        return "end_node"
    return "table_selection_node"


def table_selection_node(state: SeedGenerationState) -> SeedGenerationState:
    state["selected_table_number"] = poisson_sample(mean=5, min_c=1, max_c=10)
    state["selected_database_name"], state["database_statistics"] = database_selection(state["database_statistics"])
    if state["dialect"] == "postgresql":
        state["selected_detailed_database_schema"] = postgres_db_schema_dict[state["selected_database_name"]]
    elif state["dialect"] == "oracle":
        state["selected_detailed_database_schema"] = oracle_db_schema_dict[state["selected_database_name"]]
    else:
        raise ValueError(f"Unsupported database type: {state['dialect']}. Must be 'postgresql' or 'oracle'.")

    candidate_table_count = state["selected_table_number"] + table_selection_redundancy

    if state["dialect"] == "postgresql":
        db_schema_graph = postgres_db_schema_graph
    elif state["dialect"] == "oracle":
        db_schema_graph = oracle_db_schema_graph
    else:
        raise ValueError(f"Unsupported database type: {state['dialect']}. Must be 'postgresql' or 'oracle'.")
    
    candidate_tables = random_walk_table_selection(
        db_schema_graph=db_schema_graph,
        database_name=state["selected_database_name"],
        target_table_count=candidate_table_count
    )

    selected_tables = table_selection_agent(
        database_type=state["dialect"],
        candidate_tables=candidate_tables,
        table_schemas=state["selected_detailed_database_schema"],
        target_count=state["selected_table_number"]
    )

    state["selected_tables"] = selected_tables

    return state


def generation_node(state: SeedGenerationState) -> SeedGenerationState:
    generated_queries = generation_agent(
        database_type=state["dialect"],
        selected_tables=state["selected_tables"],
        table_schemas=state["selected_detailed_database_schema"],
        query_count=5,  # 每次生成5个查询
        epoch=state["epoch"]  # 传递epoch用于控制函数文档的显示
    )

    state["generated_plsqls"] = generated_queries
    state["correction_experience"] = [[] for _ in generated_queries] if generated_queries else []
    state["execution_info"] = [""] * len(generated_queries)

    if generated_queries:
        state["current_plsql"] = generated_queries[0]
    else:
        state["current_plsql"] = ""

    state["critical_again"] = True

    return state


def critical_node(state: SeedGenerationState) -> SeedGenerationState:
    state = critical(state)
    return state


def route_on_whether_need_correction(state: SeedGenerationState) -> str:
    if state["critical_epoch"] >= 4:
        return "ir_generation_node"
    return "correction_node"


def correction_node(state: SeedGenerationState) -> SeedGenerationState:
    need_flags = state.get("need_correction", [])
    if not isinstance(need_flags, list):
        need_flags = []

    generated_plsqls = state.get("generated_plsqls", []) or []
    execution_infos = state.get("execution_info", []) or []
    correction_histories = state.get("correction_experience", [])
    critical_epoch = state.get("critical_epoch", 0)
    dialect = state.get("dialect")
    if dialect not in {"postgresql", "oracle"}:
        raise ValueError(f"Unsupported dialect '{dialect}'. Expected 'postgresql' or 'oracle'.")

    if correction_histories is None:
        correction_histories = []
    while len(correction_histories) < len(generated_plsqls):
        correction_histories.append([])

    print("correct_epoch: ", critical_epoch)
    if critical_epoch >= 3:
        for idx, need_flag in enumerate(need_flags):
            if idx >= len(correction_histories):
                continue
            if need_flag is False and correction_histories[idx]:
                summary_record = summarize_correction_history(correction_histories[idx])
                if summary_record:
                    _append_correction_summary(summary_record, dialect)
                    correction_histories[idx] = []
        state["correction_experience"] = correction_histories
        state["critical_again"] = False
        state["current_execution_info"] = ""
        return state

    any_corrected = False
    state["current_execution_info"] = ""

    for idx, need_fix in enumerate(need_flags):
        if not need_fix:
            continue
        if idx >= len(generated_plsqls):
            continue

        current_plsql = generated_plsqls[idx]
        current_info = execution_infos[idx] if idx < len(execution_infos) else ""
        prior_records = correction_histories[idx] if idx < len(correction_histories) else []

        state["current_plsql"] = current_plsql
        state["current_execution_info"] = current_info

        corrected_plsql, correction_experience_text, combined_record = correction_agent(
            current_plsql=current_plsql,
            selected_tables=state["selected_tables"],
            selected_database_schema=state["selected_detailed_database_schema"],
            execution_info=current_info,
            critical_epoch=critical_epoch,
            prior_experiences=prior_records,
            dialect=dialect,
        )

        corrected_plsql = corrected_plsql.strip()
        generated_plsqls[idx] = corrected_plsql
        state["current_plsql"] = corrected_plsql

        if idx < len(execution_infos):
            execution_infos[idx] = ""

        if idx < len(correction_histories):
            correction_histories[idx].append(combined_record)
        else:
            correction_histories.append([combined_record])

        any_corrected = True

    state["generated_plsqls"] = generated_plsqls
    state["execution_info"] = execution_infos
    # if need_flags:
    #     state["need_correction"] = [False] * len(need_flags)

    state["correction_experience"] = correction_histories
    state["critical_again"] = any_corrected
    if not any_corrected:
        state["current_execution_info"] = ""

    return state

def ir_generation_node(state: SeedGenerationState) -> SeedGenerationState:
    """
    为seeds中的每个PL/SQL生成详细的自然语言描述（IR）
    
    输入格式: state["seeds"] = [{"plsql": str, "database_name": str, "schema": dict}, ...]
    输出格式: state["seeds"] = [{"ir": str, "plsql": str, "database_name": str, "schema": dict}, ...]
    """
    dialect = state.get("dialect", "postgresql")
    seeds = state.get("seeds", [])
    
    if not seeds:
        print("Warning: No seeds found in state")
        return state
    
    print(f"\n{'='*80}")
    print(f"【开始IR生成】共 {len(seeds)} 个 PL/SQL 需要生成IR")
    print(f"数据库类型: {dialect}")
    print(f"{'='*80}\n")
    
    # 调用IR生成Agent
    updated_seeds = ir_generation_agent(
        database_type=dialect,
        seeds=seeds
    )
    
    # 更新state
    state["seeds"] = updated_seeds
    
    return state

def alignment_node(state: SeedGenerationState) -> SeedGenerationState:
    state = alignment(state)
    return state


def init_seed_generation_graph():
    seed_generation_graph = StateGraph(SeedGenerationState)

    seed_generation_graph.add_node("entry_node", entry_node)
    seed_generation_graph.add_node("dialect_node", dialect_node)
    seed_generation_graph.add_node("management_node", management_node)
    seed_generation_graph.add_node("table_selection_node", table_selection_node)
    seed_generation_graph.add_node("generation_node", generation_node)
    seed_generation_graph.add_node("critical_node", critical_node)
    seed_generation_graph.add_node("correction_node", correction_node)
    seed_generation_graph.add_node("ir_generation_node", ir_generation_node)
    seed_generation_graph.add_node("alignment_node", alignment_node)

    seed_generation_graph.add_edge(START, "entry_node")
    seed_generation_graph.add_edge("entry_node", "dialect_node")
    seed_generation_graph.add_edge("dialect_node", "management_node")
    seed_generation_graph.add_conditional_edges(
        "management_node",
        route_on_whether_continue_generation,
        {
            "table_selection_node": "table_selection_node",
            "end_node": END,
        },
    )
    seed_generation_graph.add_edge("table_selection_node", "generation_node")
    seed_generation_graph.add_edge("generation_node", "critical_node")
    seed_generation_graph.add_conditional_edges(
        "critical_node",
        route_on_whether_need_correction,
        {
            "correction_node": "correction_node",
            "ir_generation_node": "ir_generation_node",
        },
    )
    seed_generation_graph.add_edge("correction_node", "critical_node")
    seed_generation_graph.add_edge("ir_generation_node", "alignment_node")
    seed_generation_graph.add_edge("alignment_node", "management_node")
    return seed_generation_graph.compile()