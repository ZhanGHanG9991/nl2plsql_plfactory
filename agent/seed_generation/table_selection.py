import json
from typing import List
from langchain.prompts import ChatPromptTemplate

from config.common import strong_llm_config
from util.llm_util import init_llm_with_random_provider, call_llm_with_retry
import util.postgres_util as postgres_util
import util.oracle_util as oracle_util

ALLOWED_DATABASE_TYPES = ["postgresql", "mysql", "oracle"]

table_selection_prompt = ChatPromptTemplate([
    (
        "user",
        "You are an expert in database design and semantic analysis. "
        "Given a list of candidate tables from a {database_type} database, please select {target_count} tables that are most semantically related to each other.\n"
        "The selected tables should form a coherent group that could be used together in database operations (e.g., functions, procedures, or triggers).\n\n"
        "### Candidate Tables:\n{candidate_tables}\n\n"
        "### Table Schemas:\n{table_schemas}\n\n"
        "### Requirements:\n"
        "1. Select exactly {target_count} tables from the candidates\n"
        "2. Prioritize tables with foreign key relationships or semantic connections\n"
        "3. Consider tables that might be used together in real-world scenarios\n\n"
        "### Output Format:\n"
        "Return ONLY a JSON array of the selected table names, without any additional explanations.\n"
        "Example: [\"table1\", \"table2\", \"table3\"]"
    )
])

def _call_table_selection_llm_with_retry(prompt, max_retries: int = 3, timeout: float = 120.0):
    """
    使用超时和重试机制调用表选择LLM
    
    Args:
        prompt: 要发送给LLM的prompt
        max_retries: 最大重试次数，默认3次
        timeout: 每次调用的超时时间（秒），默认120秒
    
    Returns:
        LLM响应对象
    """
    table_selection_model_cfg = strong_llm_config.get("table_selection_model", {})
    
    def llm_call(llm):
        """LLM调用函数"""
        return llm.invoke(prompt)
    
    # 使用超时重试机制调用
    response = call_llm_with_retry(
        llm_func_factory=llm_call,
        model_config=table_selection_model_cfg,
        max_retries=max_retries,
        timeout=timeout,
        model_name="table_selection_model",
        verbose=True
    )
    
    return response

def select_semantic_tables(database_type: str, candidate_tables: List[str], table_schemas: dict, target_count: int) -> List[str]:
    """
    从候选表中选择语义相关的表
    
    Args:
        database_type: 数据库类型
        candidate_tables: 候选表名列表
        table_schemas: 表结构信息
        target_count: 需要选择的表数量
    
    Returns:
        选中的表名列表
    """
    if database_type not in ALLOWED_DATABASE_TYPES:
        raise ValueError(f"Database type {database_type} is not allowed.")
    
    # 如果候选表数量小于等于目标数量，直接返回所有候选表
    if len(candidate_tables) <= target_count:
        return candidate_tables
    
    # 格式化候选表列表
    candidate_tables_str = ", ".join(sorted(candidate_tables))
    
    # 格式化表结构信息
    if database_type == "postgresql":
        table_schemas_str = postgres_util.generate_schema_prompt_from_dict(table_schemas, candidate_tables)
    elif database_type == "oracle":
        table_schemas_str = oracle_util.generate_schema_prompt_from_dict(table_schemas, candidate_tables)
    else:
        raise ValueError(f"Database type {database_type} is not allowed.")
    
    prompt = table_selection_prompt.format_messages(
        database_type=database_type,
        candidate_tables=candidate_tables_str,
        table_schemas=table_schemas_str,
        target_count=target_count
    )
 
    print("\n" + "=" * 80)
    print("【TABLE SELECTION PROMPT】")
    print("=" * 80)
    print(prompt[0].content)
    print("=" * 80 + "\n")
    
    # 使用超时重试机制调用LLM
    response = _call_table_selection_llm_with_retry(prompt, max_retries=3, timeout=120.0)
    response_content = response.content.strip()

    print("\n" + "=" * 80)
    print("【LLM RESPONSE】")
    print("=" * 80)
    print(response_content)
    print("=" * 80 + "\n")
    
    # 清理可能的 markdown 代码块标记
    if response_content.startswith("```"):
        # 移除开头的 ```json 或 ```
        lines = response_content.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        # 移除结尾的 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_content = '\n'.join(lines).strip()
    
    # 解析返回的JSON数组
    try:
        # 尝试直接解析JSON
        selected_tables = json.loads(response_content)
        
        # 验证返回的是列表
        if not isinstance(selected_tables, list):
            raise ValueError("Response is not a list")
        
        # 验证选中的表都在候选表中
        selected_tables = [t for t in selected_tables if t in candidate_tables]
        
        # 如果选中的表数量不够，补充其他候选表
        if len(selected_tables) < target_count:
            remaining_tables = [t for t in candidate_tables if t not in selected_tables]
            selected_tables.extend(remaining_tables[:target_count - len(selected_tables)])
        
        # 如果选中的表数量过多，只取前target_count个
        selected_tables = selected_tables[:target_count]
        
    except (json.JSONDecodeError, ValueError) as e:
        # 如果解析失败，返回前target_count个候选表
        print(f"Failed to parse LLM response: {e}")
        print(f"Response content: {response_content}")
        selected_tables = candidate_tables[:target_count]
    
    return selected_tables


def table_selection_agent(database_type: str, candidate_tables: List[str], table_schemas: dict, target_count: int) -> List[str]:
    """
    表选择 Agent：从 random walk 获取的候选表中，使用 LLM 选择语义相关的表
    
    Args:
        database_type: 数据库类型（postgresql, mysql, oracle）
        candidate_tables: 候选表名列表
        table_schemas: 表结构信息字典
        target_count: 需要选择的表数量
    
    Returns:
        选中的表名列表
    """
    if database_type not in ALLOWED_DATABASE_TYPES:
        raise Exception(f"Database type {database_type} is not supported")
    
    # 使用 LLM 选择语义相关的表
    selected_tables = select_semantic_tables(
        database_type=database_type,
        candidate_tables=candidate_tables,
        table_schemas=table_schemas,
        target_count=target_count
    )
    
    return selected_tables
