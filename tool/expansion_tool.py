import os
import statistics
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config.common import seed_generation_config, pg_config, oc_config, expansion_config
from tool.seed_generation_tool import (
    calculate_plsql_metrics,
    PLSQL_METRICS_TARGET_MAP,
    PLSQLMetric
)

pg_generation_seed_path = seed_generation_config["pg_generation_seed_path"]
oc_generation_seed_path = seed_generation_config["oc_generation_seed_path"]
_PG_CORRECTION_LOG_PATH = Path(expansion_config['postgres_correction_experiences_path'])
_OC_CORRECTION_LOG_PATH = Path(expansion_config['oracle_correction_experiences_path'])

# 缓存分桶结构，避免重复构建
_SEED_BUCKETS_CACHE: Dict[str, Dict] = {}

# Expansion 指标统计缓存（按数据库类型和对象类型区分）
# 键为 (database_type, object_type)，值为 Dict[metric_name, PLSQLMetric]
_EXPANSION_METRICS_CACHE: Dict[Tuple[str, str], Dict[str, PLSQLMetric]] = {}


def get_postgres_generation_seed() -> dict:
    """
    获取 PostgreSQL 的 generation seed 数据
    
    Returns:
        包含 procedure, function, trigger 三个列表的字典
        格式: {"procedure": [...], "function": [...], "trigger": [...]}
    """
    if not os.path.exists(pg_generation_seed_path):
        return {"procedure": [], "function": [], "trigger": []}
    
    with open(pg_generation_seed_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 返回原始的字典结构，保持 procedure, function, trigger 的分类
    return {
        "procedure": data.get("procedure", []),
        "function": data.get("function", []),
        "trigger": data.get("trigger", [])
    }

def get_oracle_generation_seed() -> dict:
    """
    获取 Oracle 的 generation seed 数据
    
    Returns:
        包含 procedure, function, trigger 三个列表的字典
        格式: {"procedure": [...], "function": [...], "trigger": [...]}
    """
    if not os.path.exists(oc_generation_seed_path):
        return {"procedure": [], "function": [], "trigger": []}
    
    with open(oc_generation_seed_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 返回原始的字典结构，保持 procedure, function, trigger 的分类
    return {
        "procedure": data.get("procedure", []),
        "function": data.get("function", []),
        "trigger": data.get("trigger", [])
    }


def build_seed_buckets(database_type: str, force_rebuild: bool = False) -> Dict:
    """
    构建 seed 的指标分桶结构
    
    将 seed.json 中的每个 seed 按照其 PL/SQL 代码的指标值分桶到对应的区间中。
    这样可以快速根据指标约束查询满足条件的 seeds。
    
    分桶结构：
    {
        "procedure": {
            "Number of Statements": {
                (1, 4): [seed1, seed2, ...],  # 区间 -> seeds 列表
                (5, 8): [seed3, ...],
                ...
            },
            "Number of IF Statements": { ... },
            ...
        },
        "function": { ... },
        "trigger": { ... }
    }
    
    Args:
        database_type: 数据库类型 ("postgresql" 或 "oracle")
        force_rebuild: 是否强制重新构建（忽略缓存）
    
    Returns:
        分桶后的数据结构
    """
    from tool.seed_generation_tool import (
        PLSQL_METRICS_TARGET_MAP,
        calculate_plsql_metrics
    )
    
    # 检查缓存
    if not force_rebuild and database_type in _SEED_BUCKETS_CACHE:
        return _SEED_BUCKETS_CACHE[database_type]
    
    print(f"开始构建 {database_type} 的 seed buckets...")
    
    # 选择对应的 seed 文件路径
    if database_type == "postgresql":
        seed_path = pg_generation_seed_path
    elif database_type == "oracle":
        seed_path = oc_generation_seed_path
    else:
        raise ValueError(f"不支持的数据库类型: {database_type}")
    
    # 检查文件是否存在
    if not os.path.exists(seed_path):
        print(f"警告: Seed 文件 {seed_path} 不存在")
        return {}
    
    # 读取 seed 文件
    with open(seed_path, 'r', encoding='utf-8') as f:
        seeds_data = json.load(f)
    
    # 初始化分桶结构
    buckets = {}
    
    # 遍历每种对象类型（procedure, function, trigger）
    for object_type in ["procedure", "function", "trigger"]:
        if object_type not in seeds_data:
            continue
        
        print(f"处理 {object_type} 类型的 seeds...")
        
        # 获取该对象类型的指标目标分布定义
        if object_type not in PLSQL_METRICS_TARGET_MAP:
            continue
        
        target_definition = PLSQL_METRICS_TARGET_MAP[object_type]
        
        # 初始化该对象类型的分桶结构
        buckets[object_type] = {}
        
        # 为每个指标初始化空的区间桶
        for metric_name, intervals_def in target_definition.items():
            buckets[object_type][metric_name] = {}
            
            # 为每个区间初始化空列表
            for lower, upper, prob in intervals_def:
                interval_key = (lower, upper)
                buckets[object_type][metric_name][interval_key] = []
        
        # 遍历该对象类型的所有 seeds
        seeds = seeds_data[object_type]
        processed_count = 0
        
        for seed in seeds:
            plsql_code = seed.get("plsql", "")
            if not plsql_code:
                continue
            
            # 计算该 seed 的各项指标
            try:
                metrics_values = calculate_plsql_metrics(plsql_code, database_type)
                
                # 为每个指标找到对应的区间并添加到桶中
                for metric_name, metric_value in metrics_values.items():
                    if metric_name not in buckets[object_type]:
                        continue
                    
                    # 找到值所在的区间
                    for interval_key, seeds_list in buckets[object_type][metric_name].items():
                        lower, upper = interval_key
                        if lower <= metric_value <= upper:
                            # 添加 seed 到对应的桶中
                            seeds_list.append(seed)
                            break
                
                processed_count += 1
                
            except Exception as e:
                print(f"警告: 处理 seed 时出错: {e}")
                continue
        
        print(f"  成功处理 {processed_count} 个 {object_type} seeds")
    
    # 缓存结果
    _SEED_BUCKETS_CACHE[database_type] = buckets
    
    print(f"完成 {database_type} 的 seed buckets 构建")
    return buckets


def get_seeds_by_metrics(
    database_type: str,
    object_type: str,
    metric_constraints: Dict[str, Tuple[int, int]],
    max_results: int = 10
) -> List[Dict]:
    """
    根据指标约束获取满足条件的 seeds
    
    优化策略：对各个指标约束对应的桶取交集，避免重新计算指标
    
    Args:
        database_type: 数据库类型 ("postgresql" 或 "oracle")
        object_type: 对象类型 ("procedure", "function", "trigger")
        metric_constraints: 指标约束，格式为 {metric_name: (lower, upper)}
                          例如: {
                              "Number of Statements": (5, 8),
                              "Cyclomatic Complexity": (2, 2)
                          }
        max_results: 返回的最大结果数量
    
    Returns:
        满足所有指标约束的 seeds 列表
    """
    # 获取或构建分桶结构
    buckets = build_seed_buckets(database_type)
    
    # 检查对象类型是否存在
    if object_type not in buckets:
        print(f"警告: 对象类型 {object_type} 不存在于 {database_type} 的 buckets 中")
        return []
    
    # 收集每个指标约束对应的 seeds 集合
    seed_sets = []
    seed_lists = []  # 保存原始列表，用于后续返回完整的 seed 对象
    
    for metric_name, interval in metric_constraints.items():
        # 检查指标是否存在
        if metric_name not in buckets[object_type]:
            print(f"警告: 指标 {metric_name} 不存在于 {object_type} 的 buckets 中")
            return []
        
        # 检查区间是否存在
        if interval not in buckets[object_type][metric_name]:
            print(f"警告: 区间 {interval} 不存在于 {metric_name} 的 buckets 中")
            return []
        
        # 获取该指标-区间下的 seeds
        seeds_in_interval = buckets[object_type][metric_name][interval]
        seed_lists.append(seeds_in_interval)
        
        # 使用 plsql 代码作为唯一标识（更可靠）
        # 因为同一个 seed 可能在 JSON 解析时创建了多个对象实例
        seed_set = {seed.get("plsql", "") for seed in seeds_in_interval}
        seed_sets.append(seed_set)
    
    # 如果只有一个约束，随机返回
    if len(metric_constraints) == 1:
        seeds = seed_lists[0]
        if len(seeds) <= max_results:
            return seeds
        return random.sample(seeds, max_results)
    
    # 对所有集合取交集
    common_plsql_codes = seed_sets[0]
    for seed_set in seed_sets[1:]:
        common_plsql_codes = common_plsql_codes.intersection(seed_set)
    
    # 如果交集为空，直接返回
    if not common_plsql_codes:
        return []
    
    # 从第一个约束的桶中筛选出交集中的 seeds
    result_seeds = []
    for seed in seed_lists[0]:
        plsql_code = seed.get("plsql", "")
        if plsql_code in common_plsql_codes:
            result_seeds.append(seed)
    
    # 随机返回 max_results 个 seeds
    if len(result_seeds) <= max_results:
        return result_seeds
    return random.sample(result_seeds, max_results)


def get_random_seed_by_metrics(
    database_type: str,
    object_type: str,
    metric_constraints: Dict[str, Tuple[int, int]],
    allow_relaxation: bool = True
) -> Optional[Dict]:
    """
    根据指标约束随机获取一个满足条件的 seed
    
    如果没有完全满足约束的 seed，会尝试逐步放松约束：
    1. 尝试原始约束
    2. 如果失败且约束数量 > 1，尝试逐步减少约束数量（从 n-1 到 1）
    3. 如果仍失败，返回该对象类型的任意 seed
    
    Args:
        database_type: 数据库类型 ("postgresql" 或 "oracle")
        object_type: 对象类型 ("procedure", "function", "trigger")
        metric_constraints: 指标约束，格式为 {metric_name: (lower, upper)}
        allow_relaxation: 是否允许放松约束（默认为 True）
    
    Returns:
        随机选择的满足条件的 seed，如果没有则返回 None
    """
    # 1. 首先尝试原始约束
    seeds = get_seeds_by_metrics(
        database_type=database_type,
        object_type=object_type,
        metric_constraints=metric_constraints,
        max_results=100
    )
    
    if seeds:
        print(f"✓ 找到 {len(seeds)} 个完全匹配的 seeds")
        return random.choice(seeds)
    
    if not allow_relaxation:
        print("✗ 未找到匹配的 seeds，且不允许放松约束")
        return None
    
    print(f"⚠ 未找到完全匹配的 seeds，开始尝试放松约束...")
    
    # 2. 策略1：减少约束数量
    if len(metric_constraints) > 1:
        seeds = _try_reduce_constraints(
            database_type, object_type, metric_constraints
        )
        if seeds:
            return random.choice(seeds)
    
    # 3. 最后的策略：返回该对象类型的任意 seed
    print(f"⚠ 减少约束策略失败，尝试返回任意 {object_type} seed")
    seeds = _get_any_seed_of_type(database_type, object_type)
    if seeds:
        return random.choice(seeds)
    
    print(f"✗ 无法找到任何 {object_type} 类型的 seed")
    return None


def _try_reduce_constraints(
    database_type: str,
    object_type: str,
    metric_constraints: Dict[str, Tuple[int, int]]
) -> List[Dict]:
    """
    通过减少约束数量来放松条件
    
    策略：从 n-1 个约束逐步减少到 1 个约束，尝试所有可能的组合
    """
    from itertools import combinations
    
    constraint_items = list(metric_constraints.items())
    n = len(constraint_items)
    
    # 从 n-1 个约束开始尝试，逐步减少
    for size in range(n - 1, 0, -1):
        print(f"  尝试使用 {size} 个约束...")
        
        # 尝试所有可能的约束组合
        for combo in combinations(constraint_items, size):
            relaxed_constraints = dict(combo)
            
            seeds = get_seeds_by_metrics(
                database_type=database_type,
                object_type=object_type,
                metric_constraints=relaxed_constraints,
                max_results=100
            )
            
            if seeds:
                constraint_names = list(relaxed_constraints.keys())
                print(f"  ✓ 使用约束 {constraint_names} 找到 {len(seeds)} 个 seeds")
                return seeds
    
    return []


def _get_any_seed_of_type(
    database_type: str,
    object_type: str
) -> List[Dict]:
    """
    获取指定对象类型的任意 seeds
    """
    if database_type == "postgresql":
        seed_path = pg_generation_seed_path
    elif database_type == "oracle":
        seed_path = oc_generation_seed_path
    else:
        return []
    
    if not os.path.exists(seed_path):
        return []
    
    with open(seed_path, 'r', encoding='utf-8') as f:
        seeds_data = json.load(f)
    
    seeds = seeds_data.get(object_type, [])
    
    if seeds:
        print(f"  ✓ 找到 {len(seeds)} 个 {object_type} 类型的 seeds")
        return seeds[:100]  # 返回前100个
    
    return []


def clear_seed_buckets_cache(database_type: Optional[str] = None):
    """
    清除 seed buckets 缓存
    
    Args:
        database_type: 要清除的数据库类型，如果为 None 则清除所有缓存
    """
    global _SEED_BUCKETS_CACHE
    
    if database_type is None:
        _SEED_BUCKETS_CACHE.clear()
        print("已清除所有 seed buckets 缓存")
    else:
        if database_type in _SEED_BUCKETS_CACHE:
            del _SEED_BUCKETS_CACHE[database_type]
            print(f"已清除 {database_type} 的 seed buckets 缓存")


def print_bucket_statistics(database_type: str, object_type: Optional[str] = None):
    """
    打印分桶统计信息
    
    Args:
        database_type: 数据库类型
        object_type: 对象类型，如果为 None 则打印所有对象类型的统计
    """
    buckets = build_seed_buckets(database_type)
    
    if not buckets:
        print(f"{database_type} 的 buckets 为空")
        return
    
    object_types = [object_type] if object_type else list(buckets.keys())
    
    print(f"\n{'='*60}")
    print(f"{database_type.upper()} Seed Buckets 统计")
    print(f"{'='*60}")
    
    for obj_type in object_types:
        if obj_type not in buckets:
            continue
        
        print(f"\n{obj_type.upper()}:")
        print(f"{'-'*60}")
        
        for metric_name, intervals_dict in buckets[obj_type].items():
            print(f"\n  {metric_name}:")
            
            total_seeds = 0
            for interval_key, seeds_list in sorted(intervals_dict.items()):
                lower, upper = interval_key
                count = len(seeds_list)
                total_seeds += count
                
                if count > 0:
                    print(f"    [{lower:3d}, {upper:3d}]: {count:4d} seeds")
            
            print(f"    总计: {total_seeds} seeds")
    
    print(f"\n{'='*60}\n")
    
    
def _get_correction_log_path(dialect: str) -> Path:
    if dialect == "postgresql":
        return _PG_CORRECTION_LOG_PATH
    if dialect == "oracle":
        return _OC_CORRECTION_LOG_PATH
    raise ValueError(f"Unsupported dialect '{dialect}'. Expected 'postgresql' or 'oracle'.")


def append_correction_summary(summary_record: dict, dialect: str) -> None:
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


def init_expansion_metrics_from_file(database_type: str, object_type: str) -> Dict[str, PLSQLMetric]:
    """
    从 expansion 文件初始化指标统计（按对象类型区分）
    
    Args:
        database_type: 数据库类型 ("postgresql" 或 "oracle")
        object_type: 对象类型 ("procedure", "function", "trigger")
    
    Returns:
        初始化好的指标对象（包含从 expansion 文件统计的 current_count）
    """
    # 验证对象类型
    if object_type not in PLSQL_METRICS_TARGET_MAP:
        raise ValueError(f"Unsupported object type: {object_type}. Must be one of {list(PLSQL_METRICS_TARGET_MAP.keys())}")
    
    # 根据数据库类型选择对应的 expansion 文件路径
    if database_type == "postgresql":
        expansion_path = expansion_config["postgres_expansion_path"]
    elif database_type == "oracle":
        expansion_path = expansion_config["oracle_expansion_path"]
    else:
        raise ValueError(f"Unsupported database type: {database_type}")
    
    # 根据对象类型获取对应的目标分布
    target_definition = PLSQL_METRICS_TARGET_MAP[object_type]
    
    # 创建新的 PLSQLMetric 实例
    metrics = {}
    for metric_name, intervals_def in target_definition.items():
        metrics[metric_name] = PLSQLMetric(metric_name, intervals_def)
    
    # 如果 expansion 文件存在，从中统计初始值（只统计对应对象类型的 items）
    if os.path.exists(expansion_path):
        try:
            with open(expansion_path, 'r', encoding='utf-8') as f:
                expansion_data = json.load(f)
            
            # 只统计当前对象类型的 items
            if object_type in expansion_data:
                items = expansion_data[object_type]
                
                # 统计每个 item 的指标
                for item in items:
                    plsql_code = item.get("plsql", "")
                    if plsql_code:
                        # 计算指标值
                        metrics_values = calculate_plsql_metrics(plsql_code, database_type)
                        
                        # 更新每个指标的区间计数
                        for metric_name, metric_value in metrics_values.items():
                            if metric_name in metrics:
                                metrics[metric_name].update_count(metric_value)
                
                print(f"从 {expansion_path} 加载了 {len(items)} 个 {object_type} 类型的已有 expansion items")
                print(f"已从现有 expansion items 初始化 {object_type} 的指标统计")
            else:
                print(f"Expansion 文件 {expansion_path} 中没有 {object_type} 类型的数据，从空统计开始")
        except Exception as e:
            print(f"Warning: 无法从 expansion 文件初始化统计: {e}")
            print(f"将从空统计开始")
    else:
        print(f"Expansion 文件 {expansion_path} 不存在，从空统计开始")
    
    return metrics


def get_expansion_metrics(database_type: str, object_type: str) -> Dict[str, PLSQLMetric]:
    """
    获取指定数据库类型和对象类型的 expansion 指标定义（带运行时统计）
    
    重要：此函数返回的是运行时维护的指标对象，包含：
    - target_prob: 目标分布（固定不变）
    - current_count: 当前统计（动态更新）
    
    首次调用时会从已有的 expansion 文件初始化 current_count，
    之后使用缓存机制保证返回同一组对象实例。
    
    Args:
        database_type: 数据库类型（postgresql 或 oracle）
        object_type: 对象类型 (procedure, function, trigger)
    
    Returns:
        指标名称到 PLSQLMetric 对象的映射（单例，带状态）
    """
    cache_key = (database_type, object_type)
    
    # 如果缓存中已存在，直接返回（保持状态）
    if cache_key in _EXPANSION_METRICS_CACHE:
        return _EXPANSION_METRICS_CACHE[cache_key]
    
    # 首次访问，从 expansion 文件初始化
    metrics = init_expansion_metrics_from_file(database_type, object_type)
    
    # 缓存起来
    _EXPANSION_METRICS_CACHE[cache_key] = metrics
    
    return metrics


def update_expansion_metric_statistics(plsql_code: str, database_type: str, object_type: str):
    """
    更新 expansion 指标统计信息
    
    在每次向 expansion 文件添加新的 PL/SQL 代码后，调用此方法更新统计
    
    Args:
        plsql_code: 新增的 PL/SQL 代码
        database_type: 数据库类型
        object_type: 对象类型 (procedure, function, trigger)
    """
    # 计算当前代码的各项指标
    metrics_values = calculate_plsql_metrics(plsql_code, database_type)
    
    # 获取指标定义（会自动从缓存获取或初始化）
    metrics = get_expansion_metrics(database_type, object_type)
    
    # 更新每个指标的区间计数
    for metric_name, metric_value in metrics_values.items():
        if metric_name in metrics:
            metrics[metric_name].update_count(metric_value)


def select_expansion_metrics_with_max_gap(
    database_type: str,
    object_type: str,
    k: int = 2
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    根据 expansion 缓存统计选择k个与目标分布差距最大的指标及其最大差距区间
    
    与 seed_generation_tool.select_metrics_with_max_gap 类似，
    但是基于 expansion.json 文件而不是 generation_seed.json 文件
    
    使用缓存机制，避免每次都重新读取文件和统计
    
    Args:
        database_type: 数据库类型 ("postgresql" 或 "oracle")
        object_type: 对象类型 ("procedure", "function", "trigger")
        k: 需要选择的指标数量
    
    Returns:
        包含 (指标名, (下界, 上界)) 的列表
    """
    # 获取缓存的指标统计（首次调用会从 expansion 文件初始化）
    metrics = get_expansion_metrics(database_type, object_type)
    
    # 计算总样本数（使用任意一个指标的总计数）
    total_samples = sum(
        interval.current_count 
        for interval in next(iter(metrics.values())).intervals
    )
    
    # 如果还没有样本，随机选择指标
    if total_samples == 0:
        print(f"警告: {object_type} 类型没有样本，返回随机指标")
        return _get_random_metrics(database_type, object_type, k)
    
    print(f"\n从 expansion 缓存统计到 {total_samples} 个有效样本")
    
    # 计算每个指标的最大差距
    metric_gaps = []
    for name, metric in metrics.items():
        max_gap_interval = metric.get_max_gap_interval(total_samples)
        current_prob = max_gap_interval.current_count / total_samples if total_samples > 0 else 0
        gap = max_gap_interval.target_prob - current_prob
        metric_gaps.append((gap, name, max_gap_interval))
    
    # 按差距降序排序，选择前k个
    metric_gaps.sort(reverse=True, key=lambda x: x[0])
    
    # 返回选中的指标及其最大差距区间（最多 k 个，不超过总指标数）
    num_to_select = min(k, len(metric_gaps))
    result = [(name, (interval.lower, interval.upper)) for gap, name, interval in metric_gaps[:num_to_select]]
    
    return result


def _get_random_metrics(
    database_type: str,
    object_type: str,
    k: int
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    随机选择k个指标及其随机区间（用于没有统计数据的情况）
    
    Args:
        database_type: 数据库类型
        object_type: 对象类型
        k: 需要选择的指标数量
    
    Returns:
        包含 (指标名, (下界, 上界)) 的列表
    """
    if object_type not in PLSQL_METRICS_TARGET_MAP:
        return []
    
    target_definition = PLSQL_METRICS_TARGET_MAP[object_type]
    
    # 随机选择k个指标
    metric_names = list(target_definition.keys())
    selected_names = random.sample(metric_names, min(k, len(metric_names)))
    
    result = []
    for name in selected_names:
        intervals_def = target_definition[name]
        # 随机选择一个区间
        lower, upper, prob = random.choice(intervals_def)
        result.append((name, (lower, upper)))
    
    return result
        