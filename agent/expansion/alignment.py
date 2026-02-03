import re
import json
from typing import Dict, List, Tuple
from langchain.prompts import ChatPromptTemplate
from agent.expansion.critical import ALLOWED_DATABASE_TYPES, save_to_json
from config.common import weak_llm_config, expansion_config
from state.expansion_state import ExpansionState
from util.llm_util import init_llm_with_random_provider, call_llm_with_retry

from util.postgres_util import get_plsql_type as get_plpgsql_type
from util.oracle_util import get_plsql_type as get_ocplsql_type

# 自然语言描述对 PL/SQL 的完整性验证提示模板
ir_plsql_verification_prompt = ChatPromptTemplate([
    (
        "user",
        "You are an expert in database programming languages and semantic analysis.\n"
        "Below is a natural language description and a {database_type} code implementation.\n"
        "Your task is to determine if the natural language description adequately captures the core semantics of the {database_type} code.\n\n"
        "### Natural Language Description:\n{ir_description}\n\n"
        "### {database_type} Code:\n{plsql_code}\n\n"
        "### Verification Criteria:\n"
        "Focus on whether the description captures the **core behavior** and **main logic flow**:\n"
        "1. Does the description convey the primary purpose and main operations?\n"
        "2. Does the description reflect the business intent?\n\n"
        "### Important Notes - Be Lenient:\n"
        "- **Implementation details** (variable names, exact SQL syntax, loop styles) can be omitted\n"
        "- **Different wording** is acceptable as long as the core meaning is preserved\n"
        "### Decision Rules:\n"
        "- If there is NO clear conflict between description and code, and the description correctly covers the core logic path and main effects, output **1**\n"
        "- Only output **0** if there is an obvious contradiction or the core semantics are missing (which would cause misunderstanding of the main behavior)\n"
        "- **When in doubt, prefer 1** - be generous in acceptance\n\n"
        "### Output Format:\n"
        "Your response must be wrapped in <start-result> and <end-result> tags\n\n"
        "<start-result>\n"
        "1 or 0\n"
        "Explanation: [Brief explanation of your decision]\n"
        "<end-result>\n\n"
        "Output '1' if the description adequately captures the core code semantics without major conflicts, '0' only if there are clear contradictions or missing core logic.\n"
        "Do NOT include any text before or after the result. Output ONLY the result in the specified format."
    )
])


def _call_alignment_llm_with_retry(prompt, max_retries: int = 3, timeout: float = 120.0):
    """
    使用超时和重试机制调用对齐检查LLM
    
    Args:
        prompt: 要发送给LLM的prompt
        max_retries: 最大重试次数，默认3次
        timeout: 每次调用的超时时间（秒），默认120秒
    
    Returns:
        LLM响应对象
    """
    alignment_model_cfg = weak_llm_config.get("alignment_model", {})
    
    def llm_call(llm):
        """LLM调用函数"""
        return llm.invoke(prompt)
    
    # 使用超时重试机制调用
    response = call_llm_with_retry(
        llm_func_factory=llm_call,
        model_config=alignment_model_cfg,
        max_retries=max_retries,
        timeout=timeout,
        model_name="alignment_model",
        verbose=True
    )
    
    return response


def extract_verification_result(text: str) -> Tuple[int, str]:
    """
    从LLM响应中提取验证结果
    
    Args:
        text: LLM的响应文本
        
    Returns:
        (验证结果(1或0), 解释说明)
    """
    pattern = r'<start-result>(.*?)<end-result>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not match:
        pattern = r'<start-result>(.*?)</end-result>'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not match:
        pattern = r'<start-result>(.*?)</start-result>'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        return 0, "无法解析LLM响应"
    
    content = match.group(1).strip()
    
    # 提取数字结果 (1 或 0)
    result_match = re.search(r'\b([01])\b', content)
    result = int(result_match.group(1)) if result_match else 0
    
    # 提取解释
    explanation_match = re.search(r'Explanation:\s*(.*)', content, re.DOTALL | re.IGNORECASE)
    explanation = explanation_match.group(1).strip() if explanation_match else content
    
    return result, explanation


def verify_ir_plsql_alignment(ir_description: str, plsql_code: str, database_type: str) -> Tuple[int, str]:
    """
    使用LLM验证自然语言描述是否完整描述了PL/SQL的语义
    
    Args:
        ir_description: 自然语言描述
        plsql_code: PL/SQL代码
        database_type: 数据库类型（postgresql 或 oracle）
        
    Returns:
        (验证结果(1表示完整描述，0表示不完整), 解释说明)
    """
    # 根据数据库类型调整显示名称
    db_display_name = "PL/pgSQL" if database_type == "postgresql" else "PL/SQL"
    
    prompt = ir_plsql_verification_prompt.format_messages(
        database_type=db_display_name,
        ir_description=ir_description,
        plsql_code=plsql_code
    )
    
    try:
        # 使用超时重试机制调用LLM
        response = _call_alignment_llm_with_retry(prompt, max_retries=3, timeout=120.0)
        response_content = response.content.strip()
        result, explanation = extract_verification_result(response_content)
        
        return result, explanation
    except Exception as e:
        error_msg = f"验证过程发生异常: {str(e)}"
        print(f"错误: {error_msg}")
        return 0, error_msg



def save_ir_plsqls(dialect, entries):
    if dialect not in ALLOWED_DATABASE_TYPES:
        raise Exception(f"Dialect {dialect} is not supported")
    
    if not entries:
        print("【WARNING】没有要保存的条目，跳过保存操作")
        return
    
    file_path = None
    if dialect == "postgresql":
        file_path = expansion_config["postgres_expansion_path"]
    elif dialect == "oracle":
        file_path = expansion_config["oracle_expansion_path"]

    empty_data = {
        "procedure": [],
        "function": [],
        "trigger": []
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = empty_data
    except FileNotFoundError:
        data = empty_data
    
    if not isinstance(data, dict):
        raise Exception("Invalid data format")
    
    for entry in entries:
        plsql_type = get_plpgsql_type(entry["plsql"]) if dialect == "postgresql" else get_ocplsql_type(entry["plsql"])
        data[plsql_type].append(entry)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def alignment(state: ExpansionState) -> ExpansionState:
    """
    基于语义的对齐验证函数（直接验证自然语言描述是否完整描述PL/SQL）
    
    主要流程:
    1. 对每条数据，使用大模型验证其自然语言描述是否完整描述了生成的PL/SQL
    2. 返回1表示完整描述，0表示不完整
    3. 保存通过验证（返回1）的数据
    
    Args:
        state: Expansion的状态
        
    Returns:
        更新后的状态
    """
    dialect = state["dialect"]
    
    if dialect not in ALLOWED_DATABASE_TYPES:
        raise Exception(f"Dialect {dialect} is not supported")
    
    # 检查是否有待处理的数据
    if not state.get("ir_plsqls") or len(state["ir_plsqls"]) == 0:
        print("\n" + "=" * 80)
        print("【WARNING】没有待验证的 IR-PLSQL 数据，跳过对齐验证")
        print("=" * 80)
        state["alignment_details"] = []
        return state
    
    results = []
    alignment_details = []
    
    print(f"\n开始语义对齐验证（自然语言-PL/SQL完整性检查），共 {len(state['ir_plsqls'])} 个")
    print("=" * 80)
    
    for i, item in enumerate(state["ir_plsqls"]):
        plsql_code = item["plsql"]
        print(f"\n[{i+1}/{len(state['ir_plsqls'])}] 处理第 {i+1} 个 IR-PLSQL...")
        
        # 获取该IR-PLSQL的自然语言描述
        data_description = state["ir_plsqls"][i]["ir"]
        print(f"自然语言描述: {data_description}")
        print(f"PL/SQL代码: {plsql_code}")
        
        # 使用LLM验证自然语言描述是否完整描述了PL/SQL
        print(f"验证自然语言描述是否完整覆盖PL/SQL语义...")
        verification_result, explanation = verify_ir_plsql_alignment(
            ir_description=data_description,
            plsql_code=plsql_code,
            database_type=dialect
        )
        
        if verification_result == 1:
            print(f"✓ 描述完整 (结果: 1)")
            print(f"  说明: {explanation}")
            results.append(True)
            alignment_details.append({
                "index": i,
                "success": True,
                "verification_result": 1,
                "description": data_description,
                "plsql": plsql_code[:200] + "...",
                "explanation": explanation
            })
        else:
            print(f"✗ 描述不完整 (结果: 0)，该IR-PLSQL将被丢弃")
            print(f"  说明: {explanation}")
            results.append(False)
            alignment_details.append({
                "index": i,
                "success": False,
                "verification_result": 0,
                "description": data_description,
                "plsql": plsql_code[:200] + "...",
                "explanation": explanation
            })
    
    # 统计结果
    success_count = sum(results)
    total_count = len(results)
    print("\n" + "=" * 80)
    print(f"对齐验证完成: {success_count}/{total_count} 个通过验证")
    if total_count > 0:
        print(f"通过率: {success_count/total_count*100:.1f}%")
    else:
        print(f"通过率: N/A (没有处理任何条目)")
    print("=" * 80)
    
    # 保存通过验证的IR-PLSQL数据
    ir_plsqls_data = []
    if total_count > 0:
        for i, item in enumerate(state["ir_plsqls"]):
            if i < len(results) and results[i]:
                ir_plsqls_data.append(state["ir_plsqls"][i])
    
    if ir_plsqls_data:
        save_ir_plsqls(dialect, ir_plsqls_data)
        state['current_plsql_number'] += len(ir_plsqls_data)
        print(f"\n✓ 已保存 {len(ir_plsqls_data)} 个通过验证的 NL-IR-PL/SQL 数据")
        
        # 更新通过验证的 ir_plsqls_data 的统计指标（更新 expansion 统计）
        print(f"\n{'='*80}")
        print(f"【更新 Expansion 指标统计】")
        print(f"{'='*80}")
        
        from tool.expansion_tool import update_expansion_metric_statistics
        
        for idx, ir_plsqls in enumerate(ir_plsqls_data, 1):
            try:
                # 获取对象类型
                plsql_type = get_plpgsql_type(ir_plsqls["plsql"]) if dialect == "postgresql" else get_ocplsql_type(ir_plsqls["plsql"])
                if plsql_type == 'unknown':
                    continue
                # 更新 expansion 指标统计
                update_expansion_metric_statistics(ir_plsqls["plsql"], dialect, plsql_type)
                print(f"  ✓ 已更新第 {idx} 个 NL-IR-PL/SQL 的 {plsql_type} expansion 指标统计")
            except Exception as e:
                print(f"  ✗ 更新第 {idx} 个 NL-IR-PL/SQL 的 expansion 指标统计失败: {e}")
        
        print(f"{'='*80}\n")
    else:
        print(f"\n⚠ 警告: 没有NL-IR-PL/SQL数据通过验证！")
    
    # 将对齐详情添加到状态中供后续分析
    state["alignment_details"] = alignment_details
    
    return state

