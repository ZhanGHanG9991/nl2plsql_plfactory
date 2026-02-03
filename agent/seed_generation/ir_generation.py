from typing import List
from langchain.prompts import ChatPromptTemplate

from config.common import strong_llm_config
from util.llm_util import init_llm_with_random_provider, call_llm_with_retry

ALLOWED_DATABASE_TYPES = ["postgresql", "mysql", "oracle"]


ir_generation_prompt = ChatPromptTemplate([
    (
        "user",
        "You are an expert in {database_type} database and PL/SQL programming. "
        "Given a PL/SQL code (stored procedure, function, or trigger), "
        "please generate a VERY DETAILED natural language description (IR - Intermediate Representation) "
        "that explains what this code does.\n\n"
        "### PL/SQL Code:\n```sql\n{plsql_code}\n```\n\n"
        "### Requirements:\n"
        "1. The description must be EXTREMELY DETAILED and comprehensive\n"
        "2. Include ALL parameters with their types and purposes\n"
        "3. Describe EVERY operation performed (updates, inserts, deletes, selects, etc.)\n"
        "4. Mention ALL table names, column names, and conditions used\n"
        "5. Explain any function calls or special operations (like UPPER(), LOWER(), etc.)\n"
        "6. Describe the logic flow and any conditional statements\n"
        "7. Use clear, precise technical language\n"
        "8. Start with the type of database object (procedure, function, trigger)\n"
        "9. The description must start with the word 'Write' followed by a natural-language explanation of what the PL/SQL code does, phrased as an instruction or specification.\n"
        "10. Do not include any mention of transactions, commits, rollbacks, locks, concurrency, isolation levels, or any database engine internals.\n"
        "11. Do not include any concluding or summarizing sentences. End the output immediately after the full logical description of the code.\n"
        "### Output Format:\n"
        "Output ONLY the natural language description in a single paragraph. "
        "Do NOT include any code, tags, or extra formatting. "
        "Just provide the detailed description directly.\n\n"
        "### Output Format Example 1:\n"
        "Write an Oracle PL/SQL stored procedure that takes parameters para_Name, para_Height, "
        "and para_People_ID, and updates the Name column in the people table to the uppercase version of "
        "para_Name for any row where the Height value is greater than para_Height and the People_ID matches para_People_ID.\n\n"
        "### Output Format Example 2:\n"
        "Write a PLpgSQL stored procedure that first modifies the weather table by setting min_dew_point_f to a specified value "
        "wherever mean_dew_point_f equals a given parameter, then iterates over all station records with an id higher than a provided "
        "value and, for each such station, checks if the name is absent and assigns a supplied name if so, and finally removes from the "
        "weather table any rows where mean_visibility_miles falls below a given threshold."
    )
])


def _call_ir_generation_llm_with_retry(prompt, max_retries: int = 3, timeout: float = 120.0):
    """
    使用超时和重试机制调用IR生成LLM
    
    Args:
        prompt: 要发送给LLM的prompt
        max_retries: 最大重试次数，默认3次
        timeout: 每次调用的超时时间（秒），默认120秒
    
    Returns:
        LLM响应对象
    """
    ir_model_cfg = strong_llm_config.get("ir_generation_model", {})
    
    def llm_call(llm):
        """LLM调用函数"""
        return llm.invoke(prompt)
    
    # 使用超时重试机制调用
    response = call_llm_with_retry(
        llm_func_factory=llm_call,
        model_config=ir_model_cfg,
        max_retries=max_retries,
        timeout=timeout,
        model_name="ir_generation_model",
        verbose=True
    )
    
    return response


def generate_ir_for_plsql(
    database_type: str,
    plsql_code: str
) -> str:
    """
    为单个PL/SQL代码生成详细的自然语言描述（IR）
    
    Args:
        database_type: 数据库类型（postgresql, oracle等）
        plsql_code: PL/SQL代码
    
    Returns:
        生成的自然语言描述
    """
    if database_type not in ALLOWED_DATABASE_TYPES:
        raise ValueError(f"Database type {database_type} is not allowed.")
    
    # 构建prompt
    prompt = ir_generation_prompt.format_messages(
        database_type=database_type,
        plsql_code=plsql_code
    )
    
    print("\n" + "=" * 80)
    print("【IR GENERATION PROMPT】")
    print("=" * 80)
    print(prompt[0].content)
    print("=" * 80 + "\n")
    
    # 使用超时重试机制调用LLM生成IR
    response = _call_ir_generation_llm_with_retry(prompt, max_retries=3, timeout=120.0)
    ir_description = response.content.strip()
    
    print("\n" + "=" * 80)
    print("【LLM RESPONSE - IR】")
    print("=" * 80)
    print(ir_description)
    print("=" * 80 + "\n")
    
    return ir_description


def ir_generation_agent(
    database_type: str,
    seeds: List[dict]
) -> List[dict]:
    """
    IR生成Agent：为seeds中的每个PL/SQL代码生成详细的自然语言描述
    
    Args:
        database_type: 数据库类型（postgresql, oracle等）
        seeds: 种子列表，每个元素格式为 {"plsql": str, "database_name": str, "schema": dict}
    
    Returns:
        更新后的种子列表，每个元素格式为 {"plsql": str, "ir": str, "database_name": str, "schema": dict}
    """
    if database_type not in ALLOWED_DATABASE_TYPES:
        raise Exception(f"Database type {database_type} is not supported")
    
    updated_seeds = []
    
    for idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*80}")
        print(f"【处理第 {idx}/{len(seeds)} 个 PL/SQL】")
        print(f"{'='*80}\n")
        
        plsql_code = seed.get("plsql", "")
        database_name = seed.get("database_name", "")
        tables = seed.get("tables", [])
        
        print(f"数据库名称: {database_name}")
        print(f"PL/SQL代码:\n{plsql_code}\n")
        
        try:
            # 生成IR
            ir_description = generate_ir_for_plsql(
                database_type=database_type,
                plsql_code=plsql_code
            )
            
            # 构建更新后的seed
            updated_seed = {
                "ir": ir_description,
                "plsql": plsql_code,
                "database_name": database_name,
                "tables": tables
            }
            
            updated_seeds.append(updated_seed)
            
            print(f"✓ 成功生成IR (第 {idx}/{len(seeds)} 个)")
            
        except Exception as e:
            print(f"✗ 生成IR失败 (第 {idx}/{len(seeds)} 个): {str(e)}")
            # 如果生成失败，仍然保留原始数据，但不添加IR字段
            updated_seed = {
                "ir": f"[IR generation failed: {str(e)}]",
                "plsql": plsql_code,
                "database_name": database_name,
                "tables": tables
            }
            updated_seeds.append(updated_seed)
    
    print(f"\n{'='*80}")
    print(f"【IR生成完成】")
    print(f"成功处理: {len(updated_seeds)}/{len(seeds)} 个 PL/SQL")
    print(f"{'='*80}\n")
    
    return updated_seeds

