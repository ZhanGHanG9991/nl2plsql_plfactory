import os
import statistics
import numpy as np
import json
import random
from typing import List, Dict, Tuple
from config.common import seed_generation_config, pg_config, oc_config

pg_input_path = pg_config["input_path"]
oc_input_path = oc_config["input_path"]
postgres_function_docs_path = seed_generation_config["postgres_function_docs_path"]
oracle_function_docs_path = seed_generation_config["oracle_function_docs_path"]
postgres_correction_experiences_path = seed_generation_config["postgres_correction_experiences_path"]
oracle_correction_experiences_path = seed_generation_config["oracle_correction_experiences_path"]
postgres_coreset_path = seed_generation_config["postgres_coreset_path"]
oracle_coreset_path = seed_generation_config["oracle_coreset_path"]
pg_generation_seed_path = seed_generation_config["pg_generation_seed_path"]
oc_generation_seed_path = seed_generation_config["oc_generation_seed_path"]

def poisson_sample(mean=5, min_c=1, max_c=10):
    while True:
        c = np.random.poisson(lam=mean)
        if min_c <= c <= max_c:
            return c

def get_postgres_function_docs() -> dict:
    with open(postgres_function_docs_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_oracle_function_docs() -> dict:
    with open(oracle_function_docs_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_postgres_coreset() -> dict:
    with open(postgres_coreset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_oracle_coreset() -> dict:
    with open(oracle_coreset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_postgres_correction_experiences() -> str:
    """
    读取 PostgreSQL 的纠错经验并格式化成指导文本
    使用前3个经验 + 随机3个后续经验（如果后续不足则全部使用）
    
    Returns:
        格式化后的纠错经验文本，用于指导大模型生成高质量的 PL/SQL
    """
    try:
        with open(postgres_correction_experiences_path, 'r', encoding='utf-8') as f:
            experiences = json.load(f)
        
        if not experiences:
            return "No generation guidelines available."
        
        # 选择前3个经验
        selected_experiences = experiences[:3]
        
        # 如果有超过3个经验，从剩余的中随机选择最多3个
        if len(experiences) > 3:
            remaining = experiences[3:]
            # 如果剩余数量不足3个，全部使用；否则随机选择3个
            if len(remaining) <= 3:
                selected_experiences.extend(remaining)
            else:
                selected_experiences.extend(random.sample(remaining, 3))
        
        # 打乱选中的经验顺序，增加多样性
        random.shuffle(selected_experiences)
        
        formatted_experiences = []
        formatted_experiences.append("Please follow these generation guidelines:")
        
        for idx, exp in enumerate(selected_experiences, 1):
            correction_exp = exp.get("correction_experience", "")
            if correction_exp:
                formatted_experiences.append(f"{idx}. {correction_exp}")
        
        return "\n".join(formatted_experiences)
    except Exception as e:
        print(f"Warning: Failed to load generation guidelines: {e}")
        return "No generation guidelines available."

def get_oracle_correction_experiences() -> str:
    """
    读取 Oracle 的纠错经验并格式化成指导文本
    使用前3个经验 + 随机3个后续经验（如果后续不足则全部使用）
    
    Returns:
        格式化后的纠错经验文本，用于指导大模型生成高质量的 PL/SQL
    """
    try:
        with open(oracle_correction_experiences_path, 'r', encoding='utf-8') as f:
            experiences = json.load(f)
        
        if not experiences:
            return "No generation guidelines available."
        
        # 选择前3个经验
        selected_experiences = experiences[:3]
        
        # 如果有超过3个经验，从剩余的中随机选择最多3个
        if len(experiences) > 3:
            remaining = experiences[3:]
            # 如果剩余数量不足3个，全部使用；否则随机选择3个
            if len(remaining) <= 3:
                selected_experiences.extend(remaining)
            else:
                selected_experiences.extend(random.sample(remaining, 3))
        
        # 打乱选中的经验顺序，增加多样性
        random.shuffle(selected_experiences)
        
        formatted_experiences = []
        formatted_experiences.append("Please follow these generation guidelines:")
        
        for idx, exp in enumerate(selected_experiences, 1):
            correction_exp = exp.get("correction_experience", "")
            if correction_exp:
                formatted_experiences.append(f"{idx}. {correction_exp}")
        
        return "\n".join(formatted_experiences)
    except Exception as e:
        print(f"Warning: Failed to load generation guidelines: {e}")
        return "No generation guidelines available."

def init_database_statistics(database_type: str) -> dict:
    """
    初始化数据库统计信息
    
    Args:
        database_type: 数据库类型，可选 "postgresql" 或 "oracle"
    
    Returns:
        dict: 数据库名到已生成种子数量的映射
    """
    database_statistics = {}
    
    # 根据数据库类型选择对应的路径
    if database_type == "postgresql":
        input_path = pg_input_path
        generation_seed_path = pg_generation_seed_path
    elif database_type == "oracle":
        input_path = oc_input_path
        generation_seed_path = oc_generation_seed_path
    else:
        raise ValueError(f"Unsupported database type: {database_type}. Must be 'postgresql' or 'oracle'.")
    
    # 扫描输入路径，初始化所有数据库的统计为0
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith('.sql'):
                database_name = filename[:-4]
                database_statistics[database_name] = 0
    
    # 从已生成的种子文件中统计每个数据库已生成的种子数量
    if os.path.exists(generation_seed_path):
        with open(generation_seed_path, 'r', encoding='utf-8') as f:
            generation_seeds = json.load(f)
        
        for obj_type in ["procedure", "function", "trigger"]:
            if obj_type in generation_seeds:
                seeds = generation_seeds[obj_type]
                for item in seeds:
                    database_name = item.get("database_name")
                    if database_name and database_name in database_statistics:
                        database_statistics[database_name] += 1
    
    return database_statistics


def database_selection(database_statistics: dict) -> tuple[str, dict]:
    sorted_databases = sorted(database_statistics.items(), key=lambda x: x[1])
    selected_database_name = sorted_databases[0][0]
    database_statistics[selected_database_name] += 1
    return selected_database_name, database_statistics


def random_walk_table_selection(db_schema_graph: dict, database_name: str, target_table_count: int) -> List[str]:
    """
    在数据库的外键关系图中随机游走，选择表。
    
    Args:
        db_schema_graph: 数据库外键关系图
        database_name: 数据库名
        target_table_count: 目标表数量
    
    Returns:
        选中的表名列表
    """
    # 获取指定数据库的表和外键关系
    if database_name not in db_schema_graph:
        return []
    
    db_schema = db_schema_graph[database_name]
    all_tables = db_schema.get("tables", [])
    
    if not all_tables:
        return []
    
    # 如果目标数量大于等于所有表的数量，直接返回所有表
    if target_table_count >= len(all_tables):
        return all_tables.copy()
    
    # 随机选择起点表
    selected_tables = []
    start_table = random.choice(all_tables)
    selected_tables.append(start_table)
    
    # 使用集合来快速查找已选择的表
    selected_set = {start_table}
    
    # 候选表队列（与已选表有外键关系的表）
    candidate_tables = []
    
    # 将起点表的邻接表加入候选队列
    neighbors = db_schema.get(start_table, [])
    for neighbor in neighbors:
        if neighbor not in selected_set and neighbor in all_tables:
            candidate_tables.append(neighbor)
    
    # 随机游走，直到收集到足够的表
    while len(selected_tables) < target_table_count:
        # 如果候选队列不为空，优先从候选队列中选择（保证连通性）
        if candidate_tables:
            # 随机选择一个候选表
            next_table = random.choice(candidate_tables)
            candidate_tables.remove(next_table)
        else:
            # 如果候选队列为空，从所有未选择的表中随机选择
            unselected_tables = [t for t in all_tables if t not in selected_set]
            if not unselected_tables:  # 添加安全检查
                break
            next_table = random.choice(unselected_tables)
        
        # 添加到已选择列表
        selected_tables.append(next_table)
        selected_set.add(next_table)
        
        # 将新表的邻接表加入候选队列
        neighbors = db_schema.get(next_table, [])
        for neighbor in neighbors:
            if neighbor not in selected_set and neighbor in all_tables and neighbor not in candidate_tables:
                candidate_tables.append(neighbor)
    
    return selected_tables


# ==================== PL/SQL 指标分布管理 ====================

class MetricInterval:
    """表示一个指标的取值区间"""
    def __init__(self, lower: int, upper: int, target_prob: float):
        self.lower = lower
        self.upper = upper
        self.target_prob = target_prob
        self.current_count = 0  # 当前落在该区间的样本数
    
    def contains(self, value: int) -> bool:
        """判断值是否在区间内"""
        return self.lower <= value <= self.upper
    
    def get_midpoint(self) -> float:
        """获取区间中点"""
        return (self.lower + self.upper) / 2


class PLSQLMetric:
    """表示一个 PL/SQL 统计指标及其目标分布"""
    def __init__(self, name: str, intervals: List[Tuple[int, int, float]]):
        """
        Args:
            name: 指标名称
            intervals: 区间列表，每个元素为 (lower, upper, target_prob)
        """
        self.name = name
        self.intervals = [MetricInterval(lower, upper, prob) for lower, upper, prob in intervals]
        
        # 验证概率和为1
        total_prob = sum(interval.target_prob for interval in self.intervals)
        if abs(total_prob - 1.0) > 1e-6:
            print(f"Warning: 指标 {name} 的概率和为 {total_prob}，已自动归一化")
            # 归一化
            for interval in self.intervals:
                interval.target_prob /= total_prob
    
    def get_max_gap_interval(self, total_samples: int) -> MetricInterval:
        """获取当前分布与目标分布差距最大的区间"""
        max_gap = -float('inf')
        max_gap_interval = None
        
        for interval in self.intervals:
            current_prob = interval.current_count / total_samples if total_samples > 0 else 0
            gap = interval.target_prob - current_prob  # 正值表示需要增加该区间的样本
            
            if gap > max_gap:
                max_gap = gap
                max_gap_interval = interval
        
        return max_gap_interval
    
    def update_count(self, value: int):
        """更新区间计数"""
        for interval in self.intervals:
            if interval.contains(value):
                interval.current_count += 1
                return


# PL/SQL 指标目标分布定义（按对象类型区分）
# 注意：这里只定义目标分布（target_prob），不包含运行时统计（current_count）
# 虽然PostgreSQL和Oracle的统计实现方式不同，但使用统一的目标分布
# 不同对象类型（procedure、function、trigger）有各自的指标分布

# Procedure 指标目标分布
PROCEDURE_PLSQL_METRICS_TARGET = {
    "Number of Statements": [
        (1, 4, 0.41),
        (5, 8, 0.19),
        (9, 12, 0.15),
        (13, 16, 0.06), 
        (17, 20, 0.04),
        (21, 24, 0.03),
        (25, 28, 0.04),
        (29, 35, 0.03),
        (36, 45, 0.02),
        (46, 60, 0.02),
        (61, 100, 0.01)
    ],
    "Number of IF Statements": [
        (0, 0, 0.45),
        (1, 1, 0.19),
        (2, 2, 0.10),     
        (3, 3, 0.055),     
        (4, 4, 0.04),     
        (5, 5, 0.039),   
        (6, 6, 0.039),
        (7, 7, 0.021),     
        (8, 8, 0.006),     
        (9, 9, 0.01),     
        (10, 10, 0.005),   
        (11, 11, 0.005),
        (12, 12, 0.005),
        (13, 18, 0.02),   
        (19, 25, 0.01),  
        (26, 30, 0.005)
    ],
    "Number of SET Statements": [
        (0, 0, 0.58),
        (1, 1, 0.115),
        (2, 2, 0.08),     
        (3, 3, 0.046),     
        (4, 4, 0.039),     
        (5, 5, 0.026),   
        (6, 6, 0.019),
        (7, 7, 0.017),     
        (8, 8, 0.022),     
        (9, 9, 0.013),     
        (10, 13, 0.025),
        (14, 17, 0.018)
    ],
    "Number of Parameters": [
        (0, 0, 0.05),     
        (1, 1, 0.17),     
        (2, 2, 0.24),     
        (3, 3, 0.29),     
        (4, 4, 0.15),    
        (5, 5, 0.08),    
        (6, 8, 0.02) 
    ],
    "Cyclomatic Complexity": [
        (1, 1, 0.4),
        (2, 2, 0.12),     
        (3, 3, 0.095),     
        (4, 4, 0.07),     
        (5, 5, 0.05),   
        (6, 6, 0.058),
        (7, 7, 0.04),     
        (8, 8, 0.049),     
        (9, 9, 0.02),     
        (10, 10, 0.018),
        (11, 13, 0.04),
        (14, 17, 0.03),
        (18, 22, 0.01)
    ]
}

# Function 指标目标分布
FUNCTION_PLSQL_METRICS_TARGET = {
    "Number of Statements": [
        (1, 2, 0.487),
        (3, 4, 0.2),
        (5, 6, 0.12),
        (7, 8, 0.055),
        (9, 10, 0.038),
        (11, 12, 0.03),
        (13, 17, 0.04),
        (18, 30, 0.03)
    ],
    "Number of IF Statements": [
        (0, 0, 0.631),
        (1, 1, 0.19),
        (2, 2, 0.063),     
        (3, 3, 0.035),     
        (4, 4, 0.02),     
        (5, 5, 0.021),   
        (6, 6, 0.005),
        (7, 7, 0.004),     
        (8, 8, 0.006),     
        (9, 9, 0.001),     
        (10, 10, 0.002),   
        (11, 11, 0.001),
        (12, 12, 0.001),
        (13, 18, 0.01),   
        (19, 25, 0.01)
    ],
    "Number of SET Statements": [
        (0, 0, 0.71),
        (1, 1, 0.11),
        (2, 2, 0.08),     
        (3, 3, 0.03),     
        (4, 4, 0.02),     
        (5, 5, 0.008),   
        (6, 6, 0.011),
        (7, 7, 0.011),     
        (8, 8, 0),     
        (9, 9, 0.002),     
        (10, 13, 0.013),
        (14, 17, 0.005)
    ],
    "Number of Parameters": [
        (0, 0, 0.217),     
        (1, 1, 0.393),     
        (2, 2, 0.23),     
        (3, 3, 0.1),     
        (4, 4, 0.03),    
        (5, 5, 0.02),    
        (6, 8, 0.01) 
    ],
    "Cyclomatic Complexity": [
        (1, 1, 0.61),
        (2, 2, 0.115),     
        (3, 3, 0.06),     
        (4, 4, 0.095),     
        (5, 5, 0.02),   
        (6, 6, 0.01),
        (7, 7, 0.02),     
        (8, 8, 0.01),     
        (9, 9, 0.01),     
        (10, 10, 0.01),
        (11, 13, 0.02),
        (14, 17, 0.01),
        (18, 22, 0.01)
    ]
}

# Trigger 指标目标分布
TRIGGER_PLSQL_METRICS_TARGET = {
    "Number of Statements": [
        (1, 1, 0.32),
        (2, 2, 0.137),
        (3, 3, 0.092),
        (4, 4, 0.11), 
        (5, 5, 0.093),
        (6, 6, 0.049),
        (7, 7, 0.028),
        (8, 8, 0.022),
        (9, 9, 0.012),
        (10, 10, 0.02),
        (11, 11, 0.02),
        (12, 12, 0.017),
        (13, 13, 0.02),
        (14, 17, 0.03),
        (18, 22, 0.02),
        (23, 30, 0.01),
    ],
    "Number of IF Statements": [
        (0, 0, 0.56),
        (1, 1, 0.235),
        (2, 2, 0.10),     
        (3, 3, 0.055),     
        (4, 4, 0.032),     
        (5, 5, 0.008),   
        (6, 6, 0.002),
        (7, 7, 0.001),     
        (8, 8, 0.002),     
        (9, 9, 0.002),     
        (10, 10, 0.001),   
        (11, 11, 0.001),
        (12, 15, 0.001)
    ],
    "Number of SET Statements": [
        (0, 0, 0.92),
        (1, 1, 0.045),
        (2, 2, 0.01),     
        (3, 3, 0.008),     
        (4, 4, 0.007),     
        (5, 5, 0.003),   
        (6, 6, 0.002),
        (7, 7, 0.002),     
        (8, 8, 0.001),     
        (9, 9, 0.001),     
        (10, 13, 0.001)
    ],
    "Cyclomatic Complexity": [
        (1, 1, 0.505),
        (2, 2, 0.246),     
        (3, 3, 0.1),     
        (4, 4, 0.075),     
        (5, 5, 0.04),   
        (6, 6, 0.007),
        (7, 7, 0.016),     
        (8, 8, 0.005),     
        (9, 9, 0.002),     
        (10, 10, 0),
        (11, 13, 0.002),
        (14, 17, 0.001),
        (18, 22, 0.001)
    ]
}

# 对象类型到指标目标的映射
PLSQL_METRICS_TARGET_MAP = {
    "procedure": PROCEDURE_PLSQL_METRICS_TARGET,
    "function": FUNCTION_PLSQL_METRICS_TARGET,
    "trigger": TRIGGER_PLSQL_METRICS_TARGET,
}

# 运行时统计缓存（按数据库类型和对象类型区分）
# 键为 (database_type, object_type)，值为 Dict[metric_name, PLSQLMetric]
_METRICS_CACHE: Dict[Tuple[str, str], Dict[str, PLSQLMetric]] = {}


def init_metrics_from_seed_file(database_type: str, object_type: str) -> Dict[str, PLSQLMetric]:
    """
    从已有的 seed 文件初始化指标统计（按对象类型区分）
    
    Args:
        database_type: 数据库类型
        object_type: 对象类型 (procedure, function, trigger)
    
    Returns:
        初始化好的指标对象（包含从 seed 文件统计的 current_count）
    """
    # 验证对象类型
    if object_type not in PLSQL_METRICS_TARGET_MAP:
        raise ValueError(f"Unsupported object type: {object_type}. Must be one of {list(PLSQL_METRICS_TARGET_MAP.keys())}")
    
    # 根据数据库类型选择对应的 seed 文件路径
    if database_type == "postgresql":
        seed_path = pg_generation_seed_path
    elif database_type == "oracle":
        seed_path = oc_generation_seed_path
    else:
        raise ValueError(f"Unsupported database type: {database_type}")
    
    # 根据对象类型获取对应的目标分布
    target_definition = PLSQL_METRICS_TARGET_MAP[object_type]
    
    # 创建新的 PLSQLMetric 实例
    metrics = {}
    for metric_name, intervals_def in target_definition.items():
        metrics[metric_name] = PLSQLMetric(metric_name, intervals_def)
    
    # 如果 seed 文件存在，从中统计初始值（只统计对应对象类型的 seeds）
    if os.path.exists(seed_path):
        try:
            with open(seed_path, 'r', encoding='utf-8') as f:
                generation_seeds = json.load(f)
            
            # 只统计当前对象类型的 seeds
            if object_type in generation_seeds:
                seeds = generation_seeds[object_type]
                
                # 统计每个 seed 的指标
                for seed in seeds:
                    plsql_code = seed.get("plsql", "")
                    if plsql_code:
                        # 计算指标值
                        metrics_values = calculate_plsql_metrics(plsql_code, database_type)
                        
                        # 更新每个指标的区间计数
                        for metric_name, metric_value in metrics_values.items():
                            if metric_name in metrics:
                                metrics[metric_name].update_count(metric_value)
                
                print(f"从 {seed_path} 加载了 {len(seeds)} 个 {object_type} 类型的已有 seeds")
                print(f"已从现有 seeds 初始化 {object_type} 的指标统计")
            else:
                print(f"Seed 文件 {seed_path} 中没有 {object_type} 类型的数据，从空统计开始")
        except Exception as e:
            print(f"Warning: 无法从 seed 文件初始化统计: {e}")
            print(f"将从空统计开始")
    else:
        print(f"Seed 文件 {seed_path} 不存在，从空统计开始")
    
    return metrics


def get_plsql_metrics(database_type: str, object_type: str) -> Dict[str, PLSQLMetric]:
    """
    获取指定数据库类型和对象类型的 PL/SQL 指标定义（带运行时统计）
    
    重要：此函数返回的是运行时维护的指标对象，包含：
    - target_prob: 目标分布（固定不变）
    - current_count: 当前统计（动态更新）
    
    首次调用时会从已有的 seed 文件（如 postgres_generation_seed.json）
    初始化 current_count，之后使用缓存机制保证返回同一组对象实例。
    
    Args:
        database_type: 数据库类型（postgresql 或 oracle）
        object_type: 对象类型 (procedure, function, trigger)
    
    Returns:
        指标名称到 PLSQLMetric 对象的映射（单例，带状态）
    """
    cache_key = (database_type, object_type)
    
    # 如果缓存中已存在，直接返回（保持状态）
    if cache_key in _METRICS_CACHE:
        return _METRICS_CACHE[cache_key]
    
    # 首次访问，从 seed 文件初始化
    metrics = init_metrics_from_seed_file(database_type, object_type)
    
    # 缓存起来
    _METRICS_CACHE[cache_key] = metrics
    
    return metrics


def count_statements(plsql_code: str, database_type: str = "postgresql") -> int:
    """
    统计 PL/SQL 代码中的语句总数
    
    统计规则：
    - 统计以分号结尾的语句数量
    - 排除注释中的分号
    - 排除字符串字面量中的分号
    - PostgreSQL trigger: 总数 -3
    - PostgreSQL function/procedure: 总数 -2
    - Oracle: 总数 -1
    
    Args:
        plsql_code: PL/SQL 代码
        database_type: 数据库类型
    
    Returns:
        语句总数
    """
    import re
    
    # 移除单行注释
    code = re.sub(r'--.*?$', '', plsql_code, flags=re.MULTILINE)
    
    # 移除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 移除字符串字面量
    code = re.sub(r"'[^']*'", '', code)
    
    # 统计所有分号
    statement_count = code.count(';')
    
    # 根据数据库类型和对象类型调整
    if database_type.lower() == "postgresql":
        # 检测是否是 trigger
        if re.search(r'\bCREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\b', plsql_code, re.IGNORECASE):
            statement_count -= 3
        else:
            # function 或 procedure
            statement_count -= 2
    else:  # Oracle
        statement_count -= 1
    
    return max(statement_count, 1)  # 至少有1条语句


def count_if_statements(plsql_code: str, database_type: str = "postgresql") -> int:
    """
    统计 PL/SQL 代码中的 IF 语句数量
    
    统计规则：
    - 统计 IF、ELSIF/ELSEIF 关键字出现次数
    - 忽略大小写
    - 排除注释和字符串中的关键字
    - 排除 END IF 中的 IF
    - 排除 IF EXISTS/IF NOT EXISTS (DDL 语句)
    
    Args:
        plsql_code: PL/SQL 代码
        database_type: 数据库类型
    
    Returns:
        IF 语句数量
    """
    import re
    
    # 移除单行注释
    code = re.sub(r'--.*?$', '', plsql_code, flags=re.MULTILINE)
    
    # 移除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 移除字符串字面量
    code = re.sub(r"'[^']*'", '', code)
    
    # 转换为大写以便匹配
    code_upper = code.upper()
    
    # 移除 END IF，避免误统计
    code_upper = re.sub(r'\bEND\s+IF\b', '', code_upper)
    
    # 移除 IF EXISTS 和 IF NOT EXISTS (DDL 语句中的 IF)
    code_upper = re.sub(r'\bIF\s+(?:NOT\s+)?EXISTS\b', '', code_upper)
    
    # 统计 IF 关键字（包括 ELSIF/ELSEIF）
    # 使用单词边界确保完整匹配
    if_count = len(re.findall(r'\bIF\b', code_upper))
    elsif_count = len(re.findall(r'\bELSIF\b', code_upper))
    elseif_count = len(re.findall(r'\bELSEIF\b', code_upper))
    
    total_if_count = if_count + elsif_count + elseif_count
    
    return total_if_count


def count_set_statements(plsql_code: str, database_type: str = "postgresql") -> int:
    """
    统计 PL/SQL 代码中的赋值语句数量
    
    统计规则：
    - PostgreSQL: 统计 := 赋值运算符出现次数，以及 SET 语句
    - Oracle: 统计 := 赋值运算符出现次数，以及 SET 语句
    - 排除注释和字符串中的赋值符号
    
    Args:
        plsql_code: PL/SQL 代码
        database_type: 数据库类型
    
    Returns:
        赋值语句数量
    """
    import re
    
    # 移除单行注释
    code = re.sub(r'--.*?$', '', plsql_code, flags=re.MULTILINE)
    
    # 移除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 移除字符串字面量
    code = re.sub(r"'[^']*'", '', code)
    
    # 统计 := 赋值运算符
    assignment_count = code.count(':=')
    
    # 统计 SET 语句（可能用于配置或变量赋值）
    code_upper = code.upper()
    set_count = len(re.findall(r'\bSET\b', code_upper))
    
    total_set_count = assignment_count + set_count
    
    return total_set_count


def count_table_references(plsql_code: str, database_type: str = "postgresql") -> int:
    """
    统计 PL/SQL 代码中引用的表数量
    
    统计规则：
    - 统计 FROM、JOIN、INTO、UPDATE、INSERT INTO、DELETE FROM 等子句后的表名
    - 去重，每个表只计数一次
    - 排除注释和字符串
    - 支持带双引号的标识符（如 "Table Name" 或 "schema"."table"）
    
    Args:
        plsql_code: PL/SQL 代码
        database_type: 数据库类型
    
    Returns:
        引用的表数量
    """
    import re
    
    # 移除单行注释
    code = re.sub(r'--.*?$', '', plsql_code, flags=re.MULTILINE)
    
    # 移除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 移除字符串字面量（单引号）
    code = re.sub(r"'[^']*'", '', code)
    
    # 注意：不要移除双引号，因为双引号用于标识符
    
    # 提取可能的表名
    table_names = set()
    
    # 鲁棒的标识符匹配模式：
    # - 普通标识符: [a-zA-Z_][a-zA-Z0-9_$#]*
    # - 双引号标识符: "[^"]+"
    # - 单个标识符（二选一）: (?:[a-zA-Z_][a-zA-Z0-9_$#]*|"[^"]+")
    # - 带schema前缀的完整标识符: (?:(?:[a-zA-Z_][a-zA-Z0-9_$#]*|"[^"]+")\.)?(?:[a-zA-Z_][a-zA-Z0-9_$#]*|"[^"]+")
    identifier_pattern = r'(?:(?:[a-zA-Z_][a-zA-Z0-9_$#]*|"[^"]+")\.)?(?:[a-zA-Z_][a-zA-Z0-9_$#]*|"[^"]+")'
    
    # 匹配 FROM 子句后的表名
    from_pattern = rf'\bFROM\s+({identifier_pattern})'
    table_names.update(re.findall(from_pattern, code, flags=re.IGNORECASE))
    
    # 匹配 JOIN 子句后的表名（包括 INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN 等）
    join_pattern = rf'\b(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?JOIN\s+({identifier_pattern})'
    table_names.update(re.findall(join_pattern, code, flags=re.IGNORECASE))
    
    # 匹配 UPDATE 语句的表名
    update_pattern = rf'\bUPDATE\s+({identifier_pattern})'
    table_names.update(re.findall(update_pattern, code, flags=re.IGNORECASE))
    
    # 匹配 INSERT INTO 语句的表名
    insert_pattern = rf'\bINSERT\s+INTO\s+({identifier_pattern})'
    table_names.update(re.findall(insert_pattern, code, flags=re.IGNORECASE))
    
    # 匹配 DELETE FROM 语句的表名
    delete_pattern = rf'\bDELETE\s+FROM\s+({identifier_pattern})'
    table_names.update(re.findall(delete_pattern, code, flags=re.IGNORECASE))
    
    # 匹配 INTO 子句的表名（用于 SELECT INTO）
    into_pattern = rf'\bINTO\s+({identifier_pattern})'
    # 注意：INTO 也可能用于变量，这里简化处理
    potential_tables = re.findall(into_pattern, code, flags=re.IGNORECASE)
    # 只添加看起来像表名的（通常包含点号或在特定上下文中）
    for name in potential_tables:
        if '.' in name:  # 包含schema前缀的更可能是表名
            table_names.add(name)
    
    return max(len(table_names), 1)  # 至少引用1个表


def count_parameters(plsql_code: str, database_type: str = "postgresql") -> int:
    """
    统计 PL/SQL 函数/存储过程/触发器的参数数量
    
    统计规则：
    - 提取函数/存储过程定义中的参数列表
    - Trigger Function (RETURNS TRIGGER) 不计算参数，返回 0
    - Oracle Trigger 不计算参数，返回 0
    - 统计显式参数个数
    - 支持带双引号的函数/存储过程名（如 "myFunction"）
    
    Args:
        plsql_code: PL/SQL 代码
        database_type: 数据库类型 ("postgresql" 或 "oracle")
    
    Returns:
        参数数量（Trigger 相关返回 0）
    """
    import re
    
    # 移除单行注释
    code = re.sub(r'--.*?$', '', plsql_code, flags=re.MULTILINE)
    
    # 移除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 检查是否是 Trigger Function (PostgreSQL)
    # 如果函数返回 TRIGGER 类型，则不统计参数（这些函数使用 NEW/OLD 等特殊变量）
    if re.search(r'\bRETURNS\s+TRIGGER\b', code, flags=re.IGNORECASE):
        return 0
    
    # 检查是否是 Trigger (Oracle/PostgreSQL)
    # CREATE TRIGGER ... (不是 CREATE FUNCTION)
    if re.search(r'\bCREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\b', code, flags=re.IGNORECASE):
        return 0
    
    # 鲁棒的标识符匹配模式：
    # - 普通标识符: [a-zA-Z_][a-zA-Z0-9_$#]*
    # - 双引号标识符: "[^"]+"
    # - 可能带schema前缀: (?:(?:[a-zA-Z_][a-zA-Z0-9_$#]*|"[^"]+")\.)?(?:[a-zA-Z_][a-zA-Z0-9_$#]*|"[^"]+")
    identifier_pattern = r'(?:(?:[a-zA-Z_][a-zA-Z0-9_$#]*|"[^"]+")\.)?(?:[a-zA-Z_][a-zA-Z0-9_$#]*|"[^"]+")'
    
    # 根据数据库类型使用不同的匹配策略
    if database_type.lower() == "postgresql":
        # PostgreSQL: CREATE [OR REPLACE] FUNCTION/PROCEDURE name(params) RETURNS/LANGUAGE ...
        pattern = rf'\b(?:FUNCTION|PROCEDURE)\s+({identifier_pattern})\s*\('
    else:  # Oracle
        # Oracle: CREATE [OR REPLACE] FUNCTION/PROCEDURE name(params) IS/AS/RETURN
        pattern = rf'\b(?:FUNCTION|PROCEDURE)\s+({identifier_pattern})\s*\('
    
    match = re.search(pattern, code, flags=re.IGNORECASE)
    if not match:
        return 0
    
    # 找到左括号的位置（在原始代码中而不是转换后的）
    # 重新搜索以获取准确位置
    start_pos = match.end() - 1  # 回退到左括号位置
    
    # 使用括号匹配来找到对应的右括号
    param_list = extract_parentheses_content(code, start_pos)
    
    if not param_list or not param_list.strip():
        return 0
    
    # 统计参数个数
    # 需要按逗号分割，但要注意括号嵌套（如 DEFAULT 值中的函数调用）
    param_count = count_comma_separated_items(param_list)
    
    return param_count


def extract_parentheses_content(text: str, start_pos: int) -> str:
    """
    从指定位置的左括号开始，提取匹配的括号内的内容
    
    Args:
        text: 文本内容
        start_pos: 左括号的位置
    
    Returns:
        括号内的内容（不包括括号本身）
    """
    if start_pos >= len(text) or text[start_pos] != '(':
        return ""
    
    depth = 0
    end_pos = start_pos
    
    for i in range(start_pos, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                end_pos = i
                break
    
    if depth != 0:
        # 括号不匹配
        return ""
    
    return text[start_pos + 1:end_pos]


def count_comma_separated_items(text: str) -> int:
    """
    统计逗号分隔的项目数量，考虑括号嵌套
    
    Args:
        text: 要统计的文本
    
    Returns:
        项目数量
    """
    if not text or not text.strip():
        return 0
    
    depth = 0
    count = 1  # 至少有一个参数（如果文本非空）
    
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            count += 1
    
    return count


def calculate_cyclomatic_complexity(plsql_code: str, database_type: str = "postgresql") -> int:
    """
    计算 PL/SQL 代码的圈复杂度 (Cyclomatic Complexity)
    
    圈复杂度计算公式：V(G) = E - N + 2P
    其中 E 是边数，N 是节点数，P 是连通分量数（通常为1）
    
    简化计算方式：V(G) = 1 + 决策点数量
    
    决策点包括：
    - IF, ELSIF/ELSEIF 语句
    - CASE, WHEN 语句
    - LOOP, WHILE, FOR 循环语句
    - AND, OR 逻辑运算符（在条件表达式中）
    - EXCEPTION WHEN 子句
    
    特殊处理：
    - 对于 Trigger，只计算 FUNCTION 部分的圈复杂度，不包括 CREATE TRIGGER 语句
    
    Args:
        plsql_code: PL/SQL 代码
        database_type: 数据库类型（"postgresql" 或 "oracle"）
    
    Returns:
        圈复杂度值
    """
    import re
    
    # 移除单行注释
    code = re.sub(r'--.*?$', '', plsql_code, flags=re.MULTILINE)
    
    # 移除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 移除字符串字面量
    code = re.sub(r"'[^']*'", '', code)
    
    # 转换为大写
    code_upper = code.upper()
    
    # 特殊处理：对于 Trigger，只提取 FUNCTION 部分进行计算
    # PostgreSQL Trigger 格式: CREATE FUNCTION ... RETURNS TRIGGER AS $$ ... $$ LANGUAGE plpgsql; CREATE TRIGGER ...
    # Oracle Trigger 格式: CREATE TRIGGER ... BEGIN ... END;
    if re.search(r'\bRETURNS\s+TRIGGER\b', code_upper):
        # PostgreSQL Trigger Function
        # 提取从 CREATE FUNCTION 到 $$ LANGUAGE plpgsql; 之间的部分
        function_match = re.search(
            r'(CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION.*?RETURNS\s+TRIGGER.*?\$\$.*?\$\$\s+LANGUAGE\s+PLPGSQL;?)',
            code_upper,
            re.DOTALL
        )
        if function_match:
            code_upper = function_match.group(1)
    elif re.search(r'\bCREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\b', code_upper):
        # Oracle Trigger 或 其他 Trigger 格式
        # 提取 BEGIN ... END 之间的部分（Trigger body）
        body_match = re.search(r'\bBEGIN\b(.*?)\bEND\s*;', code_upper, re.DOTALL)
        if body_match:
            code_upper = body_match.group(1)
    
    # 移除 CREATE OR REPLACE 语句中的 OR（这不是逻辑运算符）
    code_upper = re.sub(r'\bCREATE\s+OR\s+REPLACE\b', 'CREATE REPLACE', code_upper)
    
    # 移除 BETWEEN ... AND ... 中的 AND（这是 SQL 语法，不是逻辑运算符）
    # 将 BETWEEN xxx AND yyy 替换为 BETWEEN xxx TO yyy
    code_upper = re.sub(r'\bBETWEEN\s+[^)]+?\s+AND\s+', lambda m: m.group(0).replace(' AND ', ' TO '), code_upper)
    
    # 初始化复杂度为1（基础路径）
    complexity = 1
    
    # 统计 IF 和 ELSIF/ELSEIF（每个都增加一个决策点）
    # 排除 END IF
    code_temp = re.sub(r'\bEND\s+IF\b', '', code_upper)
    # 排除 IF EXISTS 和 IF NOT EXISTS (DDL 语句)
    code_temp = re.sub(r'\bIF\s+(?:NOT\s+)?EXISTS\b', '', code_temp)
    
    if_count = len(re.findall(r'\bIF\b', code_temp))
    elsif_count = len(re.findall(r'\bELSIF\b', code_temp))
    elseif_count = len(re.findall(r'\bELSEIF\b', code_temp))
    complexity += if_count + elsif_count + elseif_count
    
    # 统计 CASE 语句（CASE 本身不增加复杂度，但每个 WHEN 增加一个决策点）
    when_count = len(re.findall(r'\bWHEN\b', code_upper))
    # 排除 EXCEPTION WHEN 的 WHEN（后面会单独统计）
    exception_when_count = len(re.findall(r'\bEXCEPTION\s+WHEN\b', code_upper))
    # 排除 EXIT WHEN 和 CONTINUE WHEN 的 WHEN（这些是循环控制，不是决策点）
    exit_when_count = len(re.findall(r'\b(?:EXIT|CONTINUE)\s+WHEN\b', code_upper))
    case_when_count = when_count - exception_when_count - exit_when_count
    complexity += max(case_when_count, 0)
    
    # 统计循环语句（LOOP, WHILE, FOR）
    # FOR 和 WHILE 后面都跟着 LOOP，所以需要避免重复计数
    
    # 统计 FOR 循环（FOR ... LOOP）
    for_count = len(re.findall(r'\bFOR\b', code_upper))
    
    # 统计 WHILE 循环（WHILE ... LOOP）
    while_count = len(re.findall(r'\bWHILE\b', code_upper))
    
    # 统计独立的 LOOP（不被 FOR 或 WHILE 前导的 LOOP）
    # 总 LOOP 数 - END LOOP 数 - FOR 的 LOOP 数 - WHILE 的 LOOP 数
    loop_count = len(re.findall(r'\bLOOP\b', code_upper))
    end_loop_count = len(re.findall(r'\bEND\s+LOOP\b', code_upper))
    # FOR 和 WHILE 后面的 LOOP 不应该重复计数
    standalone_loop_count = loop_count - end_loop_count - for_count - while_count
    
    complexity += for_count + while_count + max(standalone_loop_count, 0)
    
    # 统计逻辑运算符 AND, OR（只在控制流语句中，不包括 SQL 查询条件）
    # 需要移除 SQL 语句中的 AND/OR，只保留控制流中的
    code_for_logic = code_upper
    
    # 移除 SELECT 语句（包括子查询）
    # 这会移除 SELECT ... FROM ... WHERE/JOIN/ON 中的所有 AND/OR
    code_for_logic = re.sub(
        r'\bSELECT\b.*?(?=\bINTO\b|\bFROM\b|\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|;)',
        '',
        code_for_logic,
        flags=re.DOTALL
    )
    code_for_logic = re.sub(
        r'\bFROM\b.*?(?=\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bINTO\b|;)',
        '',
        code_for_logic,
        flags=re.DOTALL
    )
    code_for_logic = re.sub(
        r'\bWHERE\b.*?(?=\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|;)',
        '',
        code_for_logic,
        flags=re.DOTALL
    )
    code_for_logic = re.sub(
        r'\bJOIN\b.*?(?=\bWHERE\b|\bJOIN\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|;)',
        '',
        code_for_logic,
        flags=re.DOTALL
    )
    
    # 移除 UPDATE/DELETE/INSERT 语句中的 WHERE 子句
    code_for_logic = re.sub(
        r'\b(?:UPDATE|DELETE|INSERT)\b.*?(?=;|\bRETURNING\b)',
        '',
        code_for_logic,
        flags=re.DOTALL
    )
    
    # 移除 MERGE 语句
    code_for_logic = re.sub(
        r'\bMERGE\b.*?(?=;)',
        '',
        code_for_logic,
        flags=re.DOTALL
    )
    
    # 现在统计剩余代码中的 AND/OR（主要是控制流语句中的）
    and_count = len(re.findall(r'\bAND\b', code_for_logic))
    or_count = len(re.findall(r'\bOR\b', code_for_logic))
    complexity += and_count + or_count
    
    # 统计异常处理 EXCEPTION WHEN
    complexity += exception_when_count
    
    # 圈复杂度至少为1
    return max(complexity, 1)


def calculate_plsql_metrics(plsql_code: str, database_type: str = "postgresql") -> Dict[str, int]:
    """
    计算 PL/SQL 代码的各项指标
    
    统计以下指标：
    - Number of Statements: 统计语句总数
    - Number of IF Statements: 统计 IF 语句数量
    - Number of SET Statements: 统计赋值语句数量
    - Number of Parameters: 统计函数/存储过程的参数数量
    - Cyclomatic Complexity: 圈复杂度
    
    Args:
        plsql_code: PL/SQL 代码
        database_type: 数据库类型
    
    Returns:
        指标名称到指标值的映射
    """
    return {
        "Number of Statements": count_statements(plsql_code, database_type),
        "Number of IF Statements": count_if_statements(plsql_code, database_type),
        "Number of SET Statements": count_set_statements(plsql_code, database_type),
        "Number of Parameters": count_parameters(plsql_code, database_type),
        "Cyclomatic Complexity": calculate_cyclomatic_complexity(plsql_code, database_type),
    }


def update_metric_statistics(plsql_code: str, database_type: str, object_type: str):
    """
    更新指标统计信息
    
    Args:
        plsql_code: 生成的 PL/SQL 代码
        database_type: 数据库类型
        object_type: 对象类型 (procedure, function, trigger)
    """
    # 计算当前代码的各项指标
    metrics_values = calculate_plsql_metrics(plsql_code, database_type)
    
    # 获取指标定义
    metrics = get_plsql_metrics(database_type, object_type)
    
    # 更新每个指标的区间计数
    for metric_name, metric_value in metrics_values.items():
        if metric_name in metrics:
            metrics[metric_name].update_count(metric_value)


def select_metrics_with_max_gap(database_type: str, object_type: str, k: int = 2) -> List[Tuple[str, MetricInterval]]:
    """
    选择k个当前分布与目标分布差距最大的指标及其最大差距区间
    
    Args:
        database_type: 数据库类型
        object_type: 对象类型 (procedure, function, trigger)
        k: 需要选择的指标数量
    
    Returns:
        包含 (指标名, 最大差距区间) 的列表
    """
    metrics = get_plsql_metrics(database_type, object_type)
    
    # 计算总样本数（使用任意一个指标的总计数）
    total_samples = sum(
        interval.current_count 
        for interval in next(iter(metrics.values())).intervals
    )
    
    # 如果还没有样本，随机选择指标
    if total_samples == 0:
        selected_metric_names = random.sample(list(metrics.keys()), min(k, len(metrics)))
        result = []
        for name in selected_metric_names:
            metric = metrics[name]
            # 随机选择一个区间
            interval = random.choice(metric.intervals)
            result.append((name, interval))
        return result
    
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
    return [(name, interval) for gap, name, interval in metric_gaps[:num_to_select]]


def format_metric_constraints(selected_metrics: List[Tuple[str, MetricInterval]]) -> str:
    """
    格式化指标约束为 prompt 文本
    
    Args:
        selected_metrics: 选中的指标及其目标区间列表
    
    Returns:
        格式化后的约束文本
    """
    if not selected_metrics:
        return "No specific metric constraints."
    
    formatted_constraints = []
    formatted_constraints.append("### IMPORTANT Metric Constraints:")
    formatted_constraints.append("The generated PL/SQL code MUST satisfy the following metric requirements:\n")
    
    for idx, (metric_name, interval) in enumerate(selected_metrics, 1):
        if interval.lower == interval.upper:
            formatted_constraints.append(
                f"{idx}. The **{metric_name}** MUST be exactly {interval.lower}."
            )
        else:
            formatted_constraints.append(
                f"{idx}. The **{metric_name}** MUST be in the range [{interval.lower}, {interval.upper}]."
            )
    
    formatted_constraints.append("\nThese constraints are CRITICAL and MUST be satisfied in your generated code.")
    
    return "\n".join(formatted_constraints)


def reset_metric_statistics(database_type: str, object_type: str = None):
    """
    重置指定数据库类型和对象类型的指标统计（清空为0）
    
    Args:
        database_type: 数据库类型
        object_type: 对象类型 (procedure, function, trigger)，如果为 None 则重置所有对象类型
    """
    if object_type is None:
        # 重置所有对象类型
        keys_to_delete = [key for key in _METRICS_CACHE.keys() if key[0] == database_type]
        for key in keys_to_delete:
            del _METRICS_CACHE[key]
        print(f"已重置 {database_type} 的所有对象类型的指标统计")
    else:
        cache_key = (database_type, object_type)
        if cache_key in _METRICS_CACHE:
            del _METRICS_CACHE[cache_key]
            print(f"已重置 {database_type} 的 {object_type} 指标统计")


def reload_metrics_from_seed_file(database_type: str, object_type: str = None):
    """
    从 seed 文件重新加载指标统计
    
    用途：当 seed 文件更新后，重新统计现有的所有 seeds
    
    Args:
        database_type: 数据库类型
        object_type: 对象类型 (procedure, function, trigger)，如果为 None 则重新加载所有对象类型
    """
    if object_type is None:
        # 重新加载所有对象类型
        for obj_type in ["procedure", "function", "trigger"]:
            cache_key = (database_type, obj_type)
            if cache_key in _METRICS_CACHE:
                del _METRICS_CACHE[cache_key]
            print(f"重新加载 {database_type} 的 {obj_type} 指标统计...")
            get_plsql_metrics(database_type, obj_type)
    else:
        # 重新加载指定对象类型
        cache_key = (database_type, object_type)
        if cache_key in _METRICS_CACHE:
            del _METRICS_CACHE[cache_key]
        print(f"重新加载 {database_type} 的 {object_type} 指标统计...")
        get_plsql_metrics(database_type, object_type)


def get_current_metric_distribution(database_type: str, object_type: str) -> Dict[str, Dict[str, any]]:
    """
    获取当前指标分布的统计信息（用于分析和可视化）
    
    Args:
        database_type: 数据库类型
        object_type: 对象类型 (procedure, function, trigger)
    
    Returns:
        包含当前分布和目标分布的详细信息
    """
    metrics = get_plsql_metrics(database_type, object_type)
    
    # 计算总样本数
    total_samples = sum(
        interval.current_count 
        for interval in next(iter(metrics.values())).intervals
    )
    
    result = {
        "total_samples": total_samples,
        "metrics": {}
    }
    
    for metric_name, metric in metrics.items():
        metric_info = {
            "intervals": [],
        }
        
        for interval in metric.intervals:
            current_prob = interval.current_count / total_samples if total_samples > 0 else 0
            gap = interval.target_prob - current_prob
            
            metric_info["intervals"].append({
                "range": f"[{interval.lower}, {interval.upper}]",
                "lower": interval.lower,
                "upper": interval.upper,
                "target_prob": interval.target_prob,
                "current_count": interval.current_count,
                "current_prob": current_prob,
                "gap": gap
            })
        
        result["metrics"][metric_name] = metric_info
    
    return result
