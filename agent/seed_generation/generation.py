import re
import random
from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
import json

from config.common import strong_llm_config
from tool.seed_generation_tool import (
    get_postgres_function_docs, 
    get_oracle_function_docs, 
    get_postgres_correction_experiences, 
    get_oracle_correction_experiences, 
    get_postgres_coreset, 
    get_oracle_coreset,
    select_metrics_with_max_gap,
    format_metric_constraints,
    calculate_plsql_metrics,
    MetricInterval,
    get_plsql_metrics
)
from util.llm_util import init_llm_with_random_provider, call_llm_with_retry
import util.postgres_util as postgres_util
import util.oracle_util as oracle_util

ALLOWED_DATABASE_TYPES = ["postgresql", "mysql", "oracle"]

# 简单内置函数分类（PostgreSQL 和 Oracle 通用）
SIMPLE_BUILTIN_FUNCTIONS = {
    "aggregate": {
        "name": "Aggregate functions",
        "examples": ["SUM()", "COUNT()", "AVG()", "MAX()", "MIN()"]
    },
    "string": {
        "name": "String functions",
        "examples": ["UPPER()", "LOWER()", "TRIM()", "LENGTH()", "SUBSTR()"]
    },
    "datetime": {
        "name": "Date/Time functions",
        "examples": ["CURRENT_DATE", "CURRENT_TIMESTAMP", "EXTRACT()"]
    },
    "mathematical": {
        "name": "Mathematical functions",
        "examples": ["ROUND()", "CEIL()", "FLOOR()", "ABS()", "MOD()"]
    },
    "conversion": {
        "name": "Conversion functions",
        "examples": ["CAST()", "TO_CHAR()", "TO_DATE()", "TO_NUMBER()"]
    }
}

# PL/SQL 对象类型定义及真实世界分布比例
PLSQL_OBJECT_TYPES = {
    "procedure": {
        "weight": 0.69,  # 69%
        "postgresql": {
            "name": "stored procedure",
            "template": "CREATE OR REPLACE PROCEDURE"
        },
        "oracle": {
            "name": "stored procedure",
            "template": "CREATE OR REPLACE PROCEDURE"
        }
    },
    "function": {
        "weight": 0.24,  # 24%
        "postgresql": {
            "name": "function",
            "template": "CREATE OR REPLACE FUNCTION"
        },
        "oracle": {
            "name": "function",
            "template": "CREATE OR REPLACE FUNCTION"
        }
    },
    "trigger": {
        "weight": 0.07,  # 7%
        "postgresql": {
            "name": "trigger",
            "template": "CREATE OR REPLACE FUNCTION <trigger_function_name>() RETURNS TRIGGER AS $$\n...\n$$ LANGUAGE plpgsql;\n\nCREATE TRIGGER <trigger_name>\n..."
        },
        "oracle": {
            "name": "trigger",
            "template": "CREATE OR REPLACE TRIGGER"
        }
    }
}

def select_plsql_object_type() -> str:
    """
    根据真实世界的比例随机选择 PL/SQL 对象类型
    
    Returns:
        选中的对象类型 ('procedure', 'function', 或 'trigger')
    """
    object_types = list(PLSQL_OBJECT_TYPES.keys())
    weights = [PLSQL_OBJECT_TYPES[obj_type]["weight"] for obj_type in object_types]
    
    # 使用权重进行随机选择
    selected_type = random.choices(object_types, weights=weights, k=1)[0]
    return selected_type


generation_prompt_zero_shot = ChatPromptTemplate([
    (
        "user",
        "You are an expert in {database_type} database and PL/SQL programming. "
        "Given a set of selected tables and their schemas, please generate a {object_type_name} that utilizes these tables.\n\n"
        "### Object Type to Generate:\n"
        "You MUST generate a **{object_type_name}**. Start your code with:\n"
        "{object_template}\n\n"
        "### Selected Tables:\n{selected_tables}\n\n"
        "### Table Schemas:\n{table_schemas}\n\n"
        "{function_section}"
        "### Generation Guidelines:\n"
        "{generation_guidelines}\n\n"
        "{metric_constraints}\n\n"
        "### Requirements:\n"
        "1. Generate {query_count} different {object_type_name}(s) in total\n"
        "{function_requirement}"
        "3. Generate queries that make use of the provided tables in meaningful ways\n"
        "4. Ensure queries are syntactically correct for {database_type}\n"
        "5. Focus on pure business logic WITHOUT any error handling\n"
        "6. Do NOT include any comments in the generated PL/SQL code\n"
        "7. Try to use different functions across different queries for diversity\n"
        "8. **IMPORTANT**: If an IF / ELSIF / ELSE structure is used, ensure that each branch performs a distinct and meaningful operation. Do NOT repeat identical or equivalent statements across multiple branches.\n"
        "9. **CRITICAL**: Your code MUST start with the template provided above: {object_template}\n"
        "10. **CRITICAL - DIVERSITY REQUIREMENT**: The {query_count} generated {object_type_name}(s) MUST be significantly different from each other in:\n"
        "    - Business logic and purpose (e.g., data aggregation vs. data transformation vs. conditional updates)\n"
        "    - Implementation approach (e.g., different SQL patterns, control flow structures, data manipulation techniques)\n"
        "    - Coding style and structure (e.g., varying use of subqueries, CTEs, joins, loops, cursors)\n"
        "    - **SQL statement types**: Balance the use of different DML operations (INSERT, UPDATE, DELETE, SELECT). Do NOT over-rely on SELECT statements; include sufficient data modification operations\n"
        "    - Do NOT generate similar or repetitive code patterns across multiple queries\n"
        "11. **CRITICAL**: Strictly adhere to the metric constraints specified above\n\n"
        "### Output Format:\n"
        "IMPORTANT: Output ONLY the queries in the following format, WITHOUT any additional explanations, descriptions, or extra text.\n"
        "Each generated query must be wrapped in <start-plsql> and <end-plsql> tags:\n\n"
        "<start-plsql>\n"
        "[Query1 here]\n"
        "<end-plsql>\n\n"
        "<start-plsql>\n"
        "[Query2 here]\n"
        "<end-plsql>\n\n"
        "Do NOT include any text before or after the queries. Output ONLY the queries in the specified format."
    )
])

generation_prompt_few_shot = ChatPromptTemplate([
    (
        "user",
        "You are an expert in {database_type} database and PL/SQL programming. "
        "Given a set of selected tables and their schemas, please generate a {object_type_name} that utilizes these tables.\n\n"
        "### Object Type to Generate:\n"
        "You MUST generate a **{object_type_name}**. Start your code with:\n"
        "{object_template}\n\n"
        "### Selected Tables:\n{selected_tables}\n\n"
        "### Table Schemas:\n{table_schemas}\n\n"
        "{function_section}"
        "### Generation Guidelines:\n"
        "{generation_guidelines}\n\n"
        "{metric_constraints}\n\n"
        "### Few-Shot Examples:\n"
        "Here are some example queries to help you understand the desired format and style:\n\n"
        "{few_shot_examples}\n\n"
        "### Requirements:\n"
        "1. Generate {query_count} different {object_type_name}(s) in total\n"
        "{function_requirement}"
        "3. Generate queries that make use of the provided tables in meaningful ways\n"
        "4. Ensure queries are syntactically correct for {database_type}\n"
        "5. Focus on pure business logic WITHOUT any error handling\n"
        "6. Do NOT include any comments in the generated PL/SQL code\n"
        "7. Try to use different functions across different queries for diversity\n"
        "8. Learn from the few-shot examples but generate NEW queries that are different from the examples\n"
        "9. **IMPORTANT**: If an IF / ELSIF / ELSE structure is used, ensure that each branch performs a distinct and meaningful operation. Do NOT repeat identical or equivalent statements across multiple branches.\n"
        "10. **CRITICAL**: Your code MUST start with the template provided above: {object_template}\n"
        "11. **CRITICAL - DIVERSITY REQUIREMENT**: The {query_count} generated {object_type_name}(s) MUST be significantly different from each other in:\n"
        "    - Business logic and purpose (e.g., data aggregation vs. data transformation vs. conditional updates)\n"
        "    - Implementation approach (e.g., different SQL patterns, control flow structures, data manipulation techniques)\n"
        "    - Coding style and structure (e.g., varying use of subqueries, CTEs, joins, loops, cursors)\n"
        "    - Complexity levels (mix simple and complex logic)\n"
        "    - **SQL statement types**: Balance the use of different DML operations (INSERT, UPDATE, DELETE, SELECT). Do NOT over-rely on SELECT statements; include sufficient data modification operations\n"
        "    - Do NOT generate similar or repetitive code patterns across multiple queries\n"
        "    - Even when learning from few-shot examples, ensure your generated PL/SQL codes are diverse and not similar to each other\n"
        "12. **CRITICAL**: Strictly adhere to the metric constraints specified above\n\n"
        "### Output Format:\n"
        "IMPORTANT: Output ONLY the queries in the following format, WITHOUT any additional explanations, descriptions, or extra text.\n"
        "Each generated query must be wrapped in <start-plsql> and <end-plsql> tags:\n\n"
        "<start-plsql>\n"
        "[Query1 here]\n"
        "<end-plsql>\n\n"
        "<start-plsql>\n"
        "[Query2 here]\n"
        "<end-plsql>\n\n"
        "Do NOT include any text before or after the queries. Output ONLY the queries in the specified format."
    )
])

# 重试生成的 Prompt（带有失败案例的详细分析和改进指导）
generation_prompt_retry = ChatPromptTemplate([
    (
        "user",
        "You are an expert in {database_type} database and PL/SQL programming. "
        "Given a set of selected tables and their schemas, please generate a {object_type_name} that utilizes these tables.\n\n"
        "{failed_examples_with_analysis}\n\n"
        "### Object Type to Generate:\n"
        "You MUST generate a **{object_type_name}**. Start your code with:\n"
        "{object_template}\n\n"
        "### Selected Tables:\n{selected_tables}\n\n"
        "### Table Schemas:\n{table_schemas}\n\n"
        "{function_section}"
        "### Generation Guidelines:\n"
        "{generation_guidelines}\n\n"
        "{metric_constraints}\n\n"
        "### Requirements:\n"
        "1. Generate {query_count} different {object_type_name}(s) in total\n"
        "{function_requirement}"
        "3. Generate queries that make use of the provided tables in meaningful ways\n"
        "4. Ensure queries are syntactically correct for {database_type}\n"
        "5. Focus on pure business logic WITHOUT any error handling\n"
        "6. Do NOT include any comments in the generated PL/SQL code\n"
        "7. Try to use different functions across different queries for diversity\n"
        "8. **IMPORTANT**: If an IF / ELSIF / ELSE structure is used, ensure that each branch performs a distinct and meaningful operation. Do NOT repeat identical or equivalent statements across multiple branches.\n"
        "9. **CRITICAL**: Your code MUST start with the template provided above: {object_template}\n"
        "10. **CRITICAL - DIVERSITY REQUIREMENT**: The {query_count} generated {object_type_name}(s) MUST be significantly different from each other in:\n"
        "    - Business logic and purpose (e.g., data aggregation vs. data transformation vs. conditional updates)\n"
        "    - Implementation approach (e.g., different SQL patterns, control flow structures, data manipulation techniques)\n"
        "    - Coding style and structure (e.g., varying use of subqueries, CTEs, joins, loops, cursors)\n"
        "    - **SQL statement types**: Balance the use of different DML operations (INSERT, UPDATE, DELETE, SELECT). Do NOT over-rely on SELECT statements; include sufficient data modification operations\n"
        "    - Do NOT generate similar or repetitive code patterns across multiple queries\n\n"
        "### 🚨🚨🚨 CRITICAL METRIC CONSTRAINTS - READ CAREFULLY 🚨🚨🚨\n"
        "{metric_constraints_emphasized}\n\n"
        "**REMINDER: The metric constraints are MANDATORY and NON-NEGOTIABLE.**\n"
        "**Your generated code will be REJECTED if it does not satisfy these constraints.**\n"
        "**Please count and verify your code structure carefully before generating!**\n\n"
        "### Output Format:\n"
        "IMPORTANT: Output ONLY the queries in the following format, WITHOUT any additional explanations, descriptions, or extra text.\n"
        "Each generated query must be wrapped in <start-plsql> and <end-plsql> tags:\n\n"
        "<start-plsql>\n"
        "[Query1 here]\n"
        "<end-plsql>\n\n"
        "<start-plsql>\n"
        "[Query2 here]\n"
        "<end-plsql>\n\n"
        "Do NOT include any text before or after the queries. Output ONLY the queries in the specified format."
    )
])

def _call_generation_llm_with_retry(prompt, max_retries: int = 3, timeout: float = 120.0):
    """
    使用超时和重试机制调用代码生成LLM
    
    Args:
        prompt: 要发送给LLM的prompt
        max_retries: 最大重试次数，默认3次
        timeout: 每次调用的超时时间（秒），默认120秒
    
    Returns:
        LLM响应对象
    """
    generation_model_cfg = strong_llm_config.get("generation_model", {})
    
    def llm_call(llm):
        """LLM调用函数"""
        return llm.invoke(prompt)
    
    # 使用超时重试机制调用
    response = call_llm_with_retry(
        llm_func_factory=llm_call,
        model_config=generation_model_cfg,
        max_retries=max_retries,
        timeout=timeout,
        model_name="generation_model",
        verbose=True
    )
    
    return response

def format_function_docs(function_docs_dict: dict, selected_functions: List[str]) -> str:
    """
    格式化函数文档为文本
    
    Args:
        function_docs_dict: 函数文档字典
        selected_functions: 选中的函数名列表
    
    Returns:
        格式化后的函数文档文本
    """
    formatted_docs = []
    
    for func_name in selected_functions:
        if func_name not in function_docs_dict:
            continue
        
        func_list = function_docs_dict[func_name]
        formatted_docs.append(f"Function: {func_name}")
        
        for idx, func_info in enumerate(func_list, 1):
            signature = func_info.get("function_signature", "")
            description = func_info.get("description", "")
            example = func_info.get("example")
            example_result = func_info.get("example_result")
            
            if len(func_list) > 1:
                formatted_docs.append(f"  Variant {idx}:")
                formatted_docs.append(f"    Signature: {signature}")
                formatted_docs.append(f"    Description: {description}")
                if example:
                    formatted_docs.append(f"    Example: {example}")
                if example_result:
                    formatted_docs.append(f"    Result: {example_result}")
            else:
                formatted_docs.append(f"  Signature: {signature}")
                formatted_docs.append(f"  Description: {description}")
                if example:
                    formatted_docs.append(f"  Example: {example}")
                if example_result:
                    formatted_docs.append(f"  Result: {example_result}")
        
        formatted_docs.append("")  # 空行分隔不同函数
    
    return "\n".join(formatted_docs)

def select_random_functions(function_docs_dict: dict, count: int = 3) -> List[str]:
    """
    从函数文档中随机选择指定数量的函数
    
    Args:
        function_docs_dict: 函数文档字典
        count: 需要选择的函数数量
    
    Returns:
        选中的函数名列表
    """
    available_functions = list(function_docs_dict.keys())
    
    # 如果可用函数数量少于请求数量，返回所有函数
    if len(available_functions) <= count:
        return available_functions
    
    # 随机选择
    return random.sample(available_functions, count)

def select_random_simple_functions(count: int = 3, examples_per_category: int = 3) -> str:
    """
    从简单内置函数中随机选择指定数量的类别，并从每个类别中随机选择函数示例
    
    Args:
        count: 需要选择的函数类别数量
        examples_per_category: 每个类别中随机选择的函数示例数量
    
    Returns:
        格式化后的函数列表文本
    """
    available_categories = list(SIMPLE_BUILTIN_FUNCTIONS.keys())
    
    # 如果可用类别数量少于请求数量，使用所有类别
    selected_count = min(count, len(available_categories))
    selected_categories = random.sample(available_categories, selected_count)
    
    formatted_lines = []
    formatted_lines.append("You are encouraged to use simple built-in functions such as:")
    
    for category_key in selected_categories:
        category = SIMPLE_BUILTIN_FUNCTIONS[category_key]
        name = category["name"]
        all_examples = category["examples"]
        
        # 从该类别的所有示例中随机选择若干个
        num_to_select = min(examples_per_category, len(all_examples))
        selected_examples = random.sample(all_examples, num_to_select)
        examples_str = ", ".join(selected_examples)
        
        formatted_lines.append(f"- **{name}**: {examples_str}")
    
    return "\n".join(formatted_lines)

def select_few_shot_examples(coreset: list, count: int = 3) -> str:
    """
    从coreset中随机选择few-shot样例并格式化
    
    Args:
        coreset: coreset列表，每个元素包含text和plsql字段
        count: 需要选择的样例数量
    
    Returns:
        格式化后的few-shot样例文本
    """
    if not coreset:
        return "No examples available."
    
    # 如果coreset数量少于请求数量，使用所有样例
    selected_count = min(count, len(coreset))
    selected_examples = random.sample(coreset, selected_count)
    
    formatted_examples = []
    for idx, example in enumerate(selected_examples, 1):
        text = example.get("text", "")
        plsql = example.get("plsql", "")
        formatted_examples.append(f"Example {idx}:")
        formatted_examples.append(f"Description: {text}")
        formatted_examples.append(f"PL/SQL Code:")
        formatted_examples.append(f"{plsql}")
        formatted_examples.append("")  # 空行分隔
    
    return "\n".join(formatted_examples)

def extract_plsql_queries(response_content: str) -> List[str]:
    """
    从 LLM 响应中提取 PL/SQL 查询
    
    Args:
        response_content: LLM 的响应内容
    
    Returns:
        提取出的查询列表
    """
    # 使用正则表达式提取所有 <start-plsql> 和 <end-plsql> 之间的内容
    pattern = r'<start-plsql>\s*(.*?)\s*<end-plsql>'
    queries = re.findall(pattern, response_content, re.DOTALL | re.IGNORECASE)
    
    # 清理每个查询的前后空白
    queries = [query.strip() for query in queries if query.strip()]
    
    return queries


def verify_metrics_constraints_flexible(
    plsql_code: str,
    database_type: str,
    object_type: str,
    selected_metrics: List[tuple]
) -> tuple:
    """
    验证生成的 PL/SQL 代码是否有助于改善选中指标的分布
    
    1. 检查实际值是否在选中的目标区间内
    2. 如果在区间内：检查该区间是否需要更多样本（gap > 0）
    3. 如果不在区间内：检查实际值所在区间是否需要样本（gap >= 0）
    
    其中 gap = target_prob - current_prob（正值表示需要更多样本）
    
    Args:
        plsql_code: PL/SQL 代码
        database_type: 数据库类型
        object_type: 对象类型
        selected_metrics: 选中的指标约束，格式为 [(metric_name, MetricInterval), ...]
    
    Returns:
        (是否接受, 详细验证结果字典)
    """
    # 计算实际指标值
    actual_metrics = calculate_plsql_metrics(plsql_code, database_type)
    
    verification_results = {}
    all_accepted = True
    
    for metric_name, target_interval in selected_metrics:
        if metric_name not in actual_metrics:
            continue
        
        actual_value = actual_metrics[metric_name]
        
        # 获取指标对象
        metrics = get_plsql_metrics(database_type, object_type)
        if metric_name not in metrics:
            continue
        
        metric = metrics[metric_name]
        
        # 计算总样本数
        total_samples = sum(interval.current_count for interval in metric.intervals)
        
        # 如果是第一个样本，总是接受
        if total_samples == 0:
            verification_results[metric_name] = {
                "actual_value": actual_value,
                "target_range": [target_interval.lower, target_interval.upper],
                "in_target_range": target_interval.contains(actual_value),
                "accepted": True,
                "reason": "If it is the first sample, always accept.",
                "selected_interval_gap": 1.0,  # 100% gap（完全缺失）
                "actual_interval_gap": None,
                "total_samples": 0
            }
            continue
        
        # 检查实际值是否在选中的目标区间内
        in_target_range = target_interval.contains(actual_value)
        
        # 计算选中区间的 gap（需求程度）
        selected_interval_current_prob = target_interval.current_count / total_samples
        selected_interval_target_prob = target_interval.target_prob
        selected_interval_gap = selected_interval_target_prob - selected_interval_current_prob
        
        # 找到实际值所属的区间
        actual_interval = None
        for interval in metric.intervals:
            if interval.contains(actual_value):
                actual_interval = interval
                break
        
        # 计算实际值所在区间的 gap
        if actual_interval:
            actual_interval_current_prob = actual_interval.current_count / total_samples
            actual_interval_target_prob = actual_interval.target_prob
            actual_interval_gap = actual_interval_target_prob - actual_interval_current_prob
        else:
            # 如果实际值不在任何定义的区间内（异常情况）
            actual_interval_gap = 0.0
        
        # 判断是否接受
        if in_target_range:
            # 实际值在选中的目标区间内
            if selected_interval_gap > 0:
                # 该区间需要更多样本
                accepted = True
                reason = f"在目标区间内，且该区间需要更多样本（gap={selected_interval_gap:.1%}）"
            else:
                # 该区间已经足够或过多
                accepted = False
                reason = f"在目标区间内，但该区间已经足够/过多（gap={selected_interval_gap:.1%}）"
        else:
            # 实际值不在选中的目标区间内
            if actual_interval_gap >= 0:
                # 实际所在区间也需要样本（或至少不过多）
                accepted = True
                reason = f"不在目标区间内，但实际区间也需要样本（gap={actual_interval_gap:.1%}）"
            else:
                # 实际所在区间已经过多
                accepted = False
                reason = f"不在目标区间内，且实际区间已过多（gap={actual_interval_gap:.1%}）"
        
        verification_results[metric_name] = {
            "actual_value": actual_value,
            "target_range": [target_interval.lower, target_interval.upper],
            "in_target_range": in_target_range,
            "accepted": accepted,
            "reason": reason,
            "selected_interval_gap": selected_interval_gap,
            "actual_interval_gap": actual_interval_gap,
            "total_samples": total_samples
        }
        
        if not accepted:
            all_accepted = False
    
    return all_accepted, verification_results


def format_failed_examples_with_analysis(
    failed_queries: List[str],
    database_type: str,
    verification_results_list: List[Dict[str, dict]],
    max_examples: int = 2
) -> str:
    """
    格式化失败的 PL/SQL 查询案例，包含代码、实际指标值和详细分析
    
    整合了原来的 verification_feedback 和 failed_examples，提供统一、清晰的反馈
    
    Args:
        failed_queries: 失败的查询列表
        database_type: 数据库类型
        verification_results_list: 每个查询对应的验证结果列表
        max_examples: 最多显示多少个失败案例（默认2个）
    
    Returns:
        格式化的失败案例文本（包含详细分析）
    """
    if not failed_queries:
        return ""
    
    feedback_lines = []
    feedback_lines.append("**🚨 YOUR PREVIOUS GENERATION FAILED - HERE'S WHAT WENT WRONG 🚨**\n")
    feedback_lines.append("I will show you the EXACT PL/SQL code you generated and explain WHY it was rejected.")
    feedback_lines.append("Study these examples carefully to understand the problem and fix it in your next generation.\n")
    
    # 只显示前 max_examples 个失败案例
    num_to_show = min(len(failed_queries), max_examples)
    
    # 使用 zip 遍历一一对应的查询和验证结果
    for i, (query, verification_results) in enumerate(zip(failed_queries[:num_to_show], 
                                                            verification_results_list[:num_to_show])):
        feedback_lines.append("=" * 80)
        feedback_lines.append(f"### ❌ FAILED EXAMPLE {i + 1} (of {len(failed_queries)} failed)")
        feedback_lines.append("=" * 80 + "\n")
        
        feedback_lines.append("**The Code You Generated:**")
        feedback_lines.append("```sql")
        feedback_lines.append(query)
        feedback_lines.append("```\n")
        
        # 显示该代码的实际指标值和详细分析
        feedback_lines.append("**Why This Code Was REJECTED:**\n")
        actual_metrics = calculate_plsql_metrics(query, database_type)
        
        has_failed_metrics = False
        for metric_name, metric_value in actual_metrics.items():
            # 如果这个指标在验证结果中，显示更详细的信息
            if metric_name in verification_results:
                result = verification_results[metric_name]
                target_range = result.get("target_range", [])
                in_target = result.get("in_target_range", False)
                accepted = result.get("accepted", False)
                reason = result.get("reason", "")
                selected_gap = result.get("selected_interval_gap", 0)
                actual_gap = result.get("actual_interval_gap")
                total_samples = result.get("total_samples", 0)
                
                status = "✅ PASS" if accepted else "❌ FAIL"
                
                # 只显示失败的指标的详细分析
                if not accepted:
                    has_failed_metrics = True
                    feedback_lines.append(f"\n{status} - **{metric_name}**")
                    feedback_lines.append(f"   📊 Your generated value: {metric_value}")
                    feedback_lines.append(f"   🎯 Required target range: [{target_range[0]}, {target_range[1]}]")
                    feedback_lines.append(f"   ❓ Is your value in target range? {'Yes ✓' if in_target else 'No ✗'}")
                    feedback_lines.append(f"   📈 Target interval gap: {selected_gap:+.1%} ({'needs MORE samples' if selected_gap > 0 else 'already ENOUGH/TOO MANY samples'})")
                    if actual_gap is not None and not in_target:
                        feedback_lines.append(f"   📉 Your value's interval gap: {actual_gap:+.1%} ({'needs more' if actual_gap > 0 else 'too many'})")
                    feedback_lines.append(f"   ⚠️  Rejection reason: {reason}")
                    feedback_lines.append(f"   📚 Current total samples in database: {total_samples}")
                else:
                    # 通过的指标只显示简单信息
                    feedback_lines.append(f"{status} - {metric_name} = {metric_value} (Target: [{target_range[0]}, {target_range[1]}])")
        
        if not has_failed_metrics:
            feedback_lines.append("(All metrics passed for this example - rejected for other reasons)")
        
        feedback_lines.append("")
    
    if len(failed_queries) > max_examples:
        feedback_lines.append(f"\n(... and {len(failed_queries) - max_examples} more failed example(s) not shown)")
    
    feedback_lines.append("\n" + "=" * 80)
    feedback_lines.append("### 💡 CRITICAL INSTRUCTIONS FOR YOUR NEXT GENERATION")
    feedback_lines.append("=" * 80 + "\n")
    feedback_lines.append("Based on the failed examples above, here's what you MUST do:\n")
    feedback_lines.append("1. **Analyze the Code Structure**: Look at how many IF statements, loops, etc. are in the failed code")
    feedback_lines.append("2. **Understand the Gap**: If gap is POSITIVE (e.g., +15%), that range NEEDS more samples → generate IN that range!")
    feedback_lines.append("3. **Understand the Gap**: If gap is NEGATIVE (e.g., -10%), that range has TOO MANY samples → AVOID that range!")
    feedback_lines.append("4. **Count Before Generating**: Mentally count your control structures BEFORE finalizing your code")
    feedback_lines.append("5. **Adjust Structure**: If you had too many IF statements, use fewer. If you had too few, use more.")
    feedback_lines.append("6. **Meet ALL Constraints**: EVERY metric constraint is MANDATORY and NON-NEGOTIABLE")
    feedback_lines.append("7. **Learn from Mistakes**: Don't repeat the same code patterns that failed above\n")
    feedback_lines.append("**Remember: Your code will be REJECTED again if you don't satisfy the metric constraints!**\n")
    
    return "\n".join(feedback_lines)

def generate_plsql_queries(
    database_type: str, 
    selected_tables: List[str], 
    table_schemas: dict, 
    query_count: int = 1,
    max_metric_retries: int = 1,
    epoch: int = 0
) -> List[str]:
    """
    生成 PL/SQL 查询（随机选择zero-shot或few-shot方式），并验证是否让分布靠近目标
    
    验证逻辑：
    - 计算加入新样本前后的分布距离
    - 如果加入后距离变小，说明让分布更接近目标，接受
    - 如果加入后距离变大，说明让分布偏离目标，拒绝并重试
    
    Args:
        database_type: 数据库类型
        selected_tables: 选中的表名列表
        table_schemas: 表结构信息
        query_count: 需要生成的查询数量
        max_metric_retries: 指标验证失败后的最大重试次数（默认1次）
        epoch: 当前的epoch数，用于控制函数文档的显示。当 epoch % 3 == 0 时提供详细的函数文档，
               其他情况鼓励使用简单的内置函数（默认为0）
    
    Returns:
        生成的查询列表
    """
    if database_type not in ALLOWED_DATABASE_TYPES:
        raise ValueError(f"Database type {database_type} is not allowed.")
    
    # 随机选择生成方式: True为few-shot(20%), False为zero-shot(80%)
    use_few_shot = random.choices([True, False], weights=[0.2, 0.8])[0]
    generation_mode = "FEW-SHOT" if use_few_shot else "ZERO-SHOT"
    
    # 随机选择 PL/SQL 对象类型
    selected_object_type = select_plsql_object_type()
    object_type_info = PLSQL_OBJECT_TYPES[selected_object_type][database_type]
    object_type_name = object_type_info["name"]
    object_template = object_type_info["template"]
    
    print(f"\n{'='*80}")
    print(f"【生成模式】: {generation_mode}")
    print(f"【对象类型】: {object_type_name} ({selected_object_type})")
    print(f"【代码模板】: {object_template}")
    print(f"{'='*80}\n")
    
    # 格式化候选表列表
    selected_tables_str = ", ".join(sorted(selected_tables))
    
    # 格式化表结构信息
    if database_type == "postgresql":
        table_schemas_str = postgres_util.generate_schema_prompt_from_dict(table_schemas, selected_tables)
    elif database_type == "oracle":
        table_schemas_str = oracle_util.generate_schema_prompt_from_dict(table_schemas, selected_tables)
    else:
        raise ValueError(f"Database type {database_type} is not allowed.")
    
    # 根据 epoch 决定是否提供详细的函数文档
    use_advanced_functions = (epoch % 3 == 0)
    
    if use_advanced_functions:
        # 每3个epoch，提供详细的函数文档
        if database_type == "postgresql":
            postgres_function_docs = get_postgres_function_docs()
            selected_functions = select_random_functions(postgres_function_docs, count=1)
            function_docs_str = format_function_docs(postgres_function_docs, selected_functions)
        elif database_type == "oracle":
            oracle_function_docs = get_oracle_function_docs()
            selected_functions = select_random_functions(oracle_function_docs, count=1)
            function_docs_str = format_function_docs(oracle_function_docs, selected_functions)
        else:
            raise ValueError(f"Unsupported database type: {database_type}. Must be 'postgresql' or 'oracle'.")
        
        function_section = (
            "### Available Functions:\n"
            "The following functions are available for you to use in your queries:\n"
            f"{function_docs_str}\n\n"
        )
        function_requirement = (
            "2. **IMPORTANT**: At least 2 of the {query_count} queries MUST use one or more functions from the 'Available Functions' list above\n"
        )
        
        print(f"\n【Epoch {epoch}】使用高级函数文档")
        print(f"随机选择的函数: {', '.join(selected_functions)}\n")
    else:
        num_categories = random.randint(2, 3)
        num_examples = random.randint(1, 4)
        selected_simple_functions = select_random_simple_functions(
            count=num_categories,
            examples_per_category=num_examples
        )
        
        function_section = (
            "### Function Usage Encouragement:\n"
            f"{selected_simple_functions}\n\n"
        )
        function_requirement = (
            "2. Feel free to use simple built-in functions (aggregates, string manipulation, date/time operations, etc.) where appropriate\n"
        )
        
        print(f"\n【Epoch {epoch}】使用简单内置函数模式")
        print(f"随机选择了 {num_categories} 个类别，每个类别 {num_examples} 个示例")
        print(f"{selected_simple_functions}\n")
    
    # 选择指标约束（选择2个差距最大的指标）
    selected_metrics = []
    try:
        selected_metrics = select_metrics_with_max_gap(database_type, selected_object_type, k=2)
        metric_constraints_str = format_metric_constraints(selected_metrics)
        
        print(f"\n{'='*80}")
        print(f"【选中的指标约束】:")
        print(f"{'='*80}")
        for metric_name, interval in selected_metrics:
            print(f"  - {metric_name}: [{interval.lower}, {interval.upper}]")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"Warning: Failed to select metric constraints: {e}")
        metric_constraints_str = ""
    
    # 获取生成指南
    try:
        if database_type == "postgresql":
            generation_guidelines_str = get_postgres_correction_experiences()
        elif database_type == "oracle":
            generation_guidelines_str = get_oracle_correction_experiences()
        else:
            generation_guidelines_str = "No generation guidelines available for this database type."
        
        print(f"\n加载的生成指南:\n{generation_guidelines_str}\n")
    except Exception as e:
        print(f"Warning: Failed to load generation guidelines: {e}")
        generation_guidelines_str = "No generation guidelines available."
    
    # 准备 few-shot 样例（如果需要）
    few_shot_examples_str = ""
    if use_few_shot:
        try:
            if database_type == "postgresql":
                coreset = get_postgres_coreset()
            elif database_type == "oracle":
                coreset = get_oracle_coreset()
            else:
                coreset = []
            
            few_shot_examples_str = select_few_shot_examples(coreset, count=3)
            print(f"\n加载的Few-Shot样例数量: 3\n")
        except Exception as e:
            print(f"Warning: Failed to load few-shot examples: {e}")
            few_shot_examples_str = "No examples available."
    
    # 生成循环：初次生成 + 可能的重试
    queries = []
    retry_count = 0
    failed_examples_with_analysis_str = ""  # 初始化失败案例分析（整合了原来的 feedback 和 examples）
    
    # 用于收集失败的查询和验证结果
    failed_queries = []
    failed_verification_results = []
    
    while retry_count <= max_metric_retries:
        is_retry = retry_count > 0
        
        if is_retry:
            print(f"\n{'🔄'*40}")
            print(f"【指标验证失败，进行第 {retry_count} 次重试】")
            print(f"{'🔄'*40}\n")
        
        # 根据是否重试选择不同的 prompt
        if is_retry:
            # 重试模式：使用带失败案例详细分析的 prompt
            prompt = generation_prompt_retry.format_messages(
                database_type=database_type,
                object_type_name=object_type_name,
                object_template=object_template,
                selected_tables=selected_tables_str,
                table_schemas=table_schemas_str,
                function_section=function_section,
                function_requirement=function_requirement,
                generation_guidelines=generation_guidelines_str,
                metric_constraints=metric_constraints_str,
                metric_constraints_emphasized=metric_constraints_str,  # 再次强调
                failed_examples_with_analysis=failed_examples_with_analysis_str,  # 统一的失败案例分析
                query_count=query_count
            )
        else:
            # 初次生成：使用标准 prompt
            if use_few_shot:
                prompt = generation_prompt_few_shot.format_messages(
                    database_type=database_type,
                    object_type_name=object_type_name,
                    object_template=object_template,
                    selected_tables=selected_tables_str,
                    table_schemas=table_schemas_str,
                    function_section=function_section,
                    function_requirement=function_requirement,
                    generation_guidelines=generation_guidelines_str,
                    metric_constraints=metric_constraints_str,
                    few_shot_examples=few_shot_examples_str,
                    query_count=query_count
                )
            else:
                prompt = generation_prompt_zero_shot.format_messages(
                    database_type=database_type,
                    object_type_name=object_type_name,
                    object_template=object_template,
                    selected_tables=selected_tables_str,
                    table_schemas=table_schemas_str,
                    function_section=function_section,
                    function_requirement=function_requirement,
                    generation_guidelines=generation_guidelines_str,
                    metric_constraints=metric_constraints_str,
                    query_count=query_count
                )
     
        print("\n" + "=" * 80)
        print("【GENERATION PROMPT】")
        print("=" * 80)
        print(prompt[0].content)
        print("=" * 80 + "\n")
        
        # 使用超时重试机制调用LLM（自动切换provider）
        response = _call_generation_llm_with_retry(
            prompt=prompt,
            max_retries=3,  # 最多重试3次
            timeout=120.0   # 每次调用超时120秒
        )
        response_content = response.content.strip()

        print("\n" + "=" * 80)
        print("【LLM RESPONSE】")
        print("=" * 80)
        print(response_content)
        print("=" * 80 + "\n")
        
        # 提取 PL/SQL 查询
        queries = extract_plsql_queries(response_content)
        
        if not queries:
            print("Warning: No queries extracted from LLM response")
            return []
        
        print(f"\n提取到 {len(queries)} 个查询")
        
        # 验证指标约束（如果有指标约束的话）
        if selected_metrics and len(queries) > 0:
            print(f"\n{'='*80}")
            print(f"【开始验证指标约束】")
            print(f"{'='*80}\n")
            
            all_queries_satisfied = True
            failed_verifications = []  # 收集所有失败的验证结果（保存查询索引和验证结果）
            
            for query_idx, query in enumerate(queries):
                # 显示时使用 query_idx + 1 让编号从1开始（更友好）
                print(f"\n--- 验证查询 {query_idx + 1}/{len(queries)} ---")
                satisfied, verification_results = verify_metrics_constraints_flexible(
                    query, 
                    database_type,
                    selected_object_type,
                    selected_metrics
                )
                
                # 打印验证结果
                for metric_name, result in verification_results.items():
                    status = "✅" if result["accepted"] else "❌"
                    in_target = "是" if result["in_target_range"] else "否"
                    selected_gap = result.get("selected_interval_gap", 0)
                    actual_gap = result.get("actual_interval_gap")
                    
                    print(f"{status} {metric_name}:")
                    print(f"   实际值={result['actual_value']}, "
                          f"目标区间=[{result['target_range'][0]}, {result['target_range'][1]}], "
                          f"在目标区间内={in_target}")
                    print(f"   目标区间gap={selected_gap:+.1%} ({'需要更多' if selected_gap > 0 else '已足够/过多'})")
                    if actual_gap is not None and not result["in_target_range"]:
                        print(f"   实际区间gap={actual_gap:+.1%} ({'需要更多' if actual_gap > 0 else '已过多'})")
                    print(f"   原因: {result['reason']}")
                
                if not satisfied:
                    all_queries_satisfied = False
                    # 保存查询索引（从0开始）和验证结果
                    failed_verifications.append((query_idx, verification_results))
            
            if all_queries_satisfied:
                print(f"\n{'✅'*40}")
                print(f"【所有查询都满足指标约束！】")
                print(f"{'✅'*40}\n")
                break  # 所有查询都满足约束，跳出循环
            else:
                print(f"\n{'❌'*40}")
                print(f"【有查询不满足指标约束】")
                print(f"{'❌'*40}\n")
                
                # 收集失败的查询和验证结果
                failed_queries.clear()
                failed_verification_results.clear()
                
                for query_idx, verification in failed_verifications:
                    # query_idx 是从 0 开始的查询索引，直接使用即可
                    failed_queries.append(queries[query_idx])
                    failed_verification_results.append(verification)
                
                # 生成统一的失败案例分析（包含代码、指标和详细分析）
                failed_examples_with_analysis_str = format_failed_examples_with_analysis(
                    failed_queries=failed_queries,
                    database_type=database_type,
                    verification_results_list=failed_verification_results,
                    max_examples=2  # 最多显示2个失败案例
                )
                
                retry_count += 1
                
                if retry_count > max_metric_retries:
                    print(f"\n{'⚠️'*40}")
                    print(f"【已达到最大重试次数 ({max_metric_retries})，返回当前生成结果】")
                    print(f"{'⚠️'*40}\n")
                    break
        else:
            # 没有指标约束，直接返回
            break
    
    return queries


def generation_agent(
    database_type: str, 
    selected_tables: List[str], 
    table_schemas: dict, 
    query_count: int = 1,
    epoch: int = 0
) -> List[str]:
    """
    生成 Agent：使用 LLM 根据选中的表生成 PL/SQL 查询
    
    Args:
        database_type: 数据库类型（postgresql, mysql, oracle）
        selected_tables: 选中的表名列表
        table_schemas: 表结构信息字典
        query_count: 需要生成的查询数量（默认为1）
        epoch: 当前的epoch数，用于控制函数文档的显示（默认为0）
    
    Returns:
        生成的查询列表
    """
    if database_type not in ALLOWED_DATABASE_TYPES:
        raise Exception(f"Database type {database_type} is not supported")
    
    # 使用 LLM 生成 PL/SQL 查询
    queries = generate_plsql_queries(
        database_type=database_type,
        selected_tables=selected_tables,
        table_schemas=table_schemas,
        query_count=query_count,
        epoch=epoch
    )
    
    return queries

