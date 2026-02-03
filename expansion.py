import json
import random
from pathlib import Path
from typing import List

from langgraph.graph import StateGraph, START, END

from state.expansion_state import ExpansionState
import util.postgres_util as postgres_util
import util.oracle_util as oracle_util
from tool.expansion_tool import get_postgres_generation_seed, get_oracle_generation_seed, append_correction_summary
from agent.expansion.seed_selection import seed_selection
from agent.expansion.ir_expansion import ir_expansion_agent
from agent.expansion.plsql_expansion import plsql_expansion_agent
from agent.expansion.critical import critical
from agent.expansion.correct import correction_agent, summarize_correction_history
from agent.expansion.alignment import alignment
from config.common import pg_config, oc_config, expansion_config


postgres_db_schema_graph = postgres_util.get_database_schema_graph()
oracle_db_schema_graph = oracle_util.get_database_schema_graph()
postgres_db_schema_dict = postgres_util.get_database_schema_json()
oracle_db_schema_dict = oracle_util.get_database_schema_json()


def start_node(state: ExpansionState) -> ExpansionState:
    state["epoch"] = 0
    state["current_plsql_number"] = 0
    
    # 判断state["dialect"]是postgresql还是oracle，加载state["seeds"]
    dialect = state.get("dialect")
    if dialect == "postgresql":
        state["seeds"] = get_postgres_generation_seed()
    elif dialect == "oracle":
        state["seeds"] = get_oracle_generation_seed()
    else:
        raise ValueError(f"Unsupported dialect '{dialect}'. Expected 'postgresql' or 'oracle'.")
    
    return state


def management_node(state: ExpansionState) -> ExpansionState:
    state["epoch"] += 1
    state["selected_seed"] = []
    return state


def route_from_management_node(state: ExpansionState) -> str:
    if state["current_plsql_number"] >= state["target_plsql_number"]:
        return "END"
    return "seed_selection_node"


def seed_selection_node(state: ExpansionState) -> ExpansionState:

    return seed_selection(
        state=state,
        postgres_db_schema_dict=postgres_db_schema_dict,
        oracle_db_schema_dict=oracle_db_schema_dict
    )


def route_from_seed_selection(state: ExpansionState) -> str:
    expansion_mode = random.choice(["ir", "plsql"])
    state["expansion_mode"] = expansion_mode
    if expansion_mode == "ir":
        return "ir_expansion_node"
    return "plsql_expansion_node"


def ir_expansion_node(state: ExpansionState) -> ExpansionState:
    selected_seed = state.get("selected_seed", []) or []
    dialect = state.get("dialect")
    if dialect not in {"postgresql", "oracle"}:
        raise ValueError(f"Unsupported dialect '{dialect}'. Expected 'postgresql' or 'oracle'.")
    if not selected_seed:
        raise ValueError("No seed data available for IR expansion.")

    expansions = ir_expansion_agent(dialect=dialect, selected_seed=selected_seed)

    state["expansion_plsqls"] = expansions
    expansion_count = len(expansions)
    state["correction_experience"] = [[] for _ in range(expansion_count)] if expansion_count else []
    state["execution_info"] = [""] * expansion_count
    state["need_correction"] = [False] * expansion_count
    state["critical_again"] = True
    state["critical_epoch"] = 0

    return state


def plsql_expansion_node(state: ExpansionState) -> ExpansionState:
    selected_seed = state.get("selected_seed", []) or []
    dialect = state.get("dialect")
    if dialect not in {"postgresql", "oracle"}:
        raise ValueError(f"Unsupported dialect '{dialect}'. Expected 'postgresql' or 'oracle'.")
    if not selected_seed:
        raise ValueError("No seed data available for PL/SQL expansion.")

    expansions = plsql_expansion_agent(dialect=dialect, selected_seed=selected_seed)

    state["expansion_plsqls"] = expansions
    expansion_count = len(expansions)
    state["correction_experience"] = [[] for _ in range(expansion_count)] if expansion_count else []
    state["execution_info"] = [""] * expansion_count
    state["need_correction"] = [False] * expansion_count
    state["critical_again"] = True
    state["critical_epoch"] = 0

    return state


def critical_node(state: ExpansionState) -> ExpansionState:
    state = critical(state)
    return state


def route_from_critical_node(state: ExpansionState) -> str:
    if state["critical_epoch"] < 5:
        return "correction_node"
    return "alignment_node"


def correction_node(state: ExpansionState) -> ExpansionState:
    need_flags = state.get("need_correction", [])
    if not isinstance(need_flags, list):
        need_flags = []

    expansion_plsql_list = [plsql["plsql"] for plsql in state.get("expansion_plsqls", [])]
    execution_infos = state.get("execution_info", []) or []
    correction_histories = state.get("correction_experience", [])
    critical_epoch = state.get("critical_epoch", 0)
    dialect = state.get("dialect")
    if dialect not in {"postgresql", "oracle"}:
        raise ValueError(f"Unsupported dialect '{dialect}'. Expected 'postgresql' or 'oracle'.")

    if correction_histories is None:
        correction_histories = []
    while len(correction_histories) < len(expansion_plsql_list):
        correction_histories.append([])

    print("correct_epoch: ", critical_epoch)
    if critical_epoch >= 4:
        for idx, need_flag in enumerate(need_flags):
            if idx >= len(correction_histories):
                continue
            if need_flag is False and correction_histories[idx]:
                summary_record = summarize_correction_history(correction_histories[idx])
                if summary_record:
                    append_correction_summary(summary_record, dialect)
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
        if idx >= len(expansion_plsql_list):
            continue

        current_plsql = expansion_plsql_list[idx]
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
        expansion_plsql_list[idx] = corrected_plsql
        state["current_plsql"] = corrected_plsql

        if idx < len(execution_infos):
            execution_infos[idx] = ""

        if idx < len(correction_histories):
            correction_histories[idx].append(combined_record)
        else:
            correction_histories.append([combined_record])

        any_corrected = True

    for i, item in enumerate(state["expansion_plsqls"]):
        item["plsql"] = expansion_plsql_list[i]
        
    state["execution_info"] = execution_infos
    # if need_flags:
    #     state["need_correction"] = [False] * len(need_flags)

    state["correction_experience"] = correction_histories
    state["critical_again"] = any_corrected
    if not any_corrected:
        state["current_execution_info"] = ""

    return state


def alignment_node(state: ExpansionState) -> ExpansionState:
    state = alignment(state)
    return state


def init_expansion_graph():
    expansion_graph = StateGraph(ExpansionState)

    expansion_graph.add_node("start_node", start_node)
    expansion_graph.add_node("management_node", management_node)
    expansion_graph.add_node("seed_selection_node", seed_selection_node)
    expansion_graph.add_node("ir_expansion_node", ir_expansion_node)
    expansion_graph.add_node("plsql_expansion_node", plsql_expansion_node)
    expansion_graph.add_node("critical_node", critical_node)
    expansion_graph.add_node("correction_node", correction_node)
    expansion_graph.add_node("alignment_node", alignment_node)

    expansion_graph.add_edge(START, "start_node")
    expansion_graph.add_edge("start_node", "management_node")
    expansion_graph.add_conditional_edges(
        "management_node",
        route_from_management_node,
        {
            "seed_selection_node": "seed_selection_node",
            "END": END,
        },
    )
    expansion_graph.add_conditional_edges(
        "seed_selection_node",
        route_from_seed_selection,
        {
            "ir_expansion_node": "ir_expansion_node",
            "plsql_expansion_node": "plsql_expansion_node",
        },
    )
    expansion_graph.add_edge("ir_expansion_node", "critical_node")
    expansion_graph.add_edge("plsql_expansion_node", "critical_node")
    expansion_graph.add_conditional_edges(
        "critical_node",
        route_from_critical_node,
        {
            "correction_node": "correction_node",
            "alignment_node": "alignment_node",
        },
    )
    expansion_graph.add_edge("correction_node", "critical_node")
    expansion_graph.add_edge("alignment_node", "management_node")

    return expansion_graph.compile()