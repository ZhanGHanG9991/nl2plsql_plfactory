import random
from typing import Dict

from state.expansion_state import ExpansionState
from tool.expansion_tool import (
    select_expansion_metrics_with_max_gap,
    get_random_seed_by_metrics
)


def seed_selection(
    state: ExpansionState,
    postgres_db_schema_dict: Dict,
    oracle_db_schema_dict: Dict
) -> ExpansionState:
    """
    基于指标差距选择 seed
    
    选择策略：
    1. 根据 PLSQL_OBJECT_TYPES 中定义的 weight 比例选择对象类型
    2. 根据 expansion 文件统计当前分布，选择差距最大的k个指标
    3. 使用 get_random_seed_by_metrics 根据指标约束选择 seed
    
    Args:
        state: 扩展状态对象
        postgres_db_schema_dict: PostgreSQL 数据库 schema 字典
        oracle_db_schema_dict: Oracle 数据库 schema 字典
    
    Returns:
        更新后的状态对象
    """
    from agent.seed_generation.generation import PLSQL_OBJECT_TYPES
    
    seeds_dict = state.get("seeds", {})
    dialect = state.get("dialect")
    
    # 按照权重选择对象类型
    object_types = list(PLSQL_OBJECT_TYPES.keys())  # ["procedure", "function", "trigger"]
    weights = [PLSQL_OBJECT_TYPES[obj_type]["weight"] for obj_type in object_types]
    
    # 根据权重随机选择对象类型
    selected_type = random.choices(object_types, weights=weights, k=1)[0]
    
    print("\n" + "=" * 80)
    print("【SEED SELECTION】")
    print("=" * 80)
    print(f"选择的对象类型: {selected_type}")
    
    # 选择指标约束（选择2个差距最大的指标）
    try:
        selected_metrics = select_expansion_metrics_with_max_gap(
            database_type=dialect,
            object_type=selected_type,
            k=2
        )
        
        print(f"\n【选中的指标约束】:")
        for metric_name, (lower, upper) in selected_metrics:
            print(f"  - {metric_name}: [{lower}, {upper}]")
        
        # 将指标格式转换为 get_random_seed_by_metrics 需要的格式
        metric_constraints = {
            metric_name: (lower, upper)
            for metric_name, (lower, upper) in selected_metrics
        }
        
        # 使用指标约束选择 seed
        selected_seed = get_random_seed_by_metrics(
            database_type=dialect,
            object_type=selected_type,
            metric_constraints=metric_constraints,
            allow_relaxation=True
        )
        
        if selected_seed is None:
            # 如果没有找到符合条件的 seed，回退到随机选择
            print(f"警告: 未找到符合指标约束的 seed，回退到随机选择")
            if seeds_dict[selected_type]:
                selected_seed = random.choice(seeds_dict[selected_type])
            else:
                raise ValueError(f"No seeds available for type {selected_type}")
        else:
            print(f"✓ 成功找到符合指标约束的 seed")
    
    except Exception as e:
        print(f"警告: 指标选择失败 ({e})，回退到随机选择")
        # 回退到随机选择
        if seeds_dict[selected_type]:
            selected_seed = random.choice(seeds_dict[selected_type])
        else:
            raise ValueError(f"No seeds available for type {selected_type}")
    
    state["selected_seed"] = [selected_seed]
    
    print("\n【选中的 Seed】:")
    print(f"  数据库: {selected_seed.get('database_name', 'N/A')}")
    print(f"  表数量: {len(selected_seed.get('tables', []))}")
    print("=" * 80 + "\n")

    state["selected_database_name"] = state["selected_seed"][0]["database_name"]
    state["selected_tables"] = state["selected_seed"][0]["tables"]

    if state["dialect"] == "postgresql":
        state["selected_detailed_database_schema"] = postgres_db_schema_dict[state["selected_database_name"]]
    elif state["dialect"] == "oracle":
        state["selected_detailed_database_schema"] = oracle_db_schema_dict[state["selected_database_name"]]
    else:
        raise ValueError(f"Unsupported database type: {state['dialect']}. Must be 'postgresql' or 'oracle'.")
    
    return state

