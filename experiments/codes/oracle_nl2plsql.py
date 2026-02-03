import os
import json
import re
import sys
import argparse
from langchain.prompts import ChatPromptTemplate

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(experiments_dir)
sys.path.append(project_root)

# 导入实验配置和工具
from experiments.settings.expr_setting import (
    get_dataset_config, 
    get_model_config, 
    get_output_path,
    LLM_CALL_CONFIG,
    is_local_model_path,
    extract_model_name_from_path,
    load_local_model,
    generate_with_local_model
)
from experiments.utils.llm_helper import call_llm_with_retry
from experiments.utils import oracle_util

# ==========================================
# 1. Oracle特定的Prompt格式化函数
# ==========================================
def format_prompt_for_local_model(databases_schema_info: str, ir_text: str) -> str:
    """
    为本地模型格式化prompt
    
    Args:
        databases_schema_info: 数据库schema信息
        ir_text: IR文本
        
    Returns:
        格式化的prompt字符串
    """
    prompt = (
            f"You are an expert in Oracle database and PL/SQL programming. Please generate an Oracle PL/SQL code "
            f"based on the provided database schema information and following the natural language instruction.\n\n"
            f"Schema Info:\n{databases_schema_info}\n\n"
            f"Instruction:\n{ir_text}\n\n"
            f"Please generate the Oracle PL/SQL code based on the schema and instruction.\n"
            f"IMPORTANT: You MUST wrap your final PL/SQL code strictly within <start-plsql> and <end-plsql> tags. Do not generate any comments or any other extraneous information; output only PL/SQL.\n"
            f"Your response:"
        )
    
    return prompt


# ==========================================
# 2. 数据加载模块
# ==========================================
def load_dataset(dataset_path: str) -> list:
    """
    加载数据集文件（数据集为简单数组格式）
    
    Args:
        dataset_path: 数据集文件路径
        
    Returns:
        数据集列表
    """
    if not os.path.exists(dataset_path):
        print(f"警告: 数据集文件不存在 {dataset_path}")
        return []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 数据集应该是一个数组
    if not isinstance(data, list):
        print(f"警告: 数据集格式不正确，应该是数组格式")
        return []
    
    return data


# ==========================================
# 3. 代码提取模块
# ==========================================
def extract_plsql_from_tags(text: str) -> str:
    """
    鲁棒地从 <start-plsql> 标记中提取代码。
    兼容以下结束标记:
    1. <end-plsql>  (标准)
    2. </start-plsql> (HTML闭合风格)
    3. </end-plsql> (混合风格)
    """
    if not text:
        return ""
    
    # --- 策略 1: 标签提取 (Tag Extraction) ---
    # 改进点: 使用原始字符串,简化转义
    tag_pattern = r"<start-plsql>\s*(.*?)\s*(?:</end-plsql>|</start-plsql>|<end-plsql>)"
    match = re.search(tag_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # --- 策略 2: Markdown 兜底 (Markdown Fallback) ---
    # 改进点: 更精确的语言标识符匹配
    md_pattern = r"```(?:sql|plsql|pl/sql)?\s*\n?(.*?)```"
    match_md = re.search(md_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match_md:
        return match_md.group(1).strip()
    
    # --- 策略 3: 最后的倔强 (Raw Strip) ---
    return text.strip()


# ==========================================
# 4. Prompt 定义 (Zero Shot)
# ==========================================
ORACLE_ZERO_SHOT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in Oracle database and PL/SQL programming. "
            "Your task is to generate valid PL/SQL code based on the provided Intermediate Representation (IR) and Database Schema."
        ),
        (
            "user",
            (
                "Database Context: Oracle\n"
                "Schema Information (Tables and Columns):\n{databases_schema_info}\n\n"
                "Intermediate Representation (IR) describing the logic:\n{ir_text}\n\n"
                "Please generate the corresponding Oracle PL/SQL code.\n"
                "IMPORTANT: You MUST wrap your final PL/SQL code strictly within <start-plsql> and <end-plsql> tags. Do not generate any comments or any other extraneous information; output only PL/SQL.\n"
                "Your response:"
            )
        ),
    ]
)


# ==========================================
# 5. 主逻辑
# ==========================================
def estimate_max_new_tokens(
    gold_plsql: str, 
    buffer_ratio: float = 1.5, 
    min_tokens: int = 256, 
    max_tokens: int = 1596,
    tokenizer=None
) -> int:
    """
    根据gold_plsql估算合理的max_new_tokens
    
    Args:
        gold_plsql: 标准答案的PL/SQL代码
        buffer_ratio: 缓冲比例，预测结果可能比gold稍长 (默认1.5x)
        min_tokens: 最小token数，防止过小 (默认256)
        max_tokens: 最大token数上限 (默认1596)
        tokenizer: 可选的tokenizer，如果提供则使用精确计算
        
    Returns:
        估算的max_new_tokens
    """
    if not gold_plsql:
        return min_tokens
    
    # 方法1: 如果有 tokenizer，使用精确计算（耗时约1-5ms，可忽略）
    if tokenizer is not None:
        try:
            token_ids = tokenizer.encode(gold_plsql, add_special_tokens=False)
            token_count = len(token_ids)
            estimated_tokens = int(token_count * buffer_ratio)
            return max(min_tokens, min(estimated_tokens, max_tokens))
        except Exception:
            pass  # 回退到字符估算
    
    # 方法2: 基于字符数估算（无tokenizer时的备选方案）
    # 对于代码类文本，平均每 3 个字符对应 1 个 token
    char_count = len(gold_plsql)
    estimated_tokens = int((char_count / 3.0) * buffer_ratio)
    
    return max(min_tokens, min(estimated_tokens, max_tokens))


def run_oracle_zero_shot_experiment(dataset_name: str, model_name: str, max_tokens: int = 8192, input_key: str = "ir"):
    """
    运行 Oracle Zero-Shot 实验
    
    Args:
        dataset_name: 数据集名称
        model_name: 模型名称或本地模型路径
        max_tokens: 最大token数
        input_key: 输入文本的字段名 (ir/summary/natural_language)
    """
    print("=" * 60)
    print("Oracle Zero-Shot NL2PL/SQL 实验")
    print("=" * 60)
    print(f"数据集: {dataset_name}")
    print(f"模型: {model_name}")
    print(f"输入字段: {input_key}")
    print("=" * 60)
    
    # 1. 判断是API模型还是本地模型
    is_local = is_local_model_path(model_name)
    
    if is_local:
        print(f"检测到本地模型路径")
        # 提取模型名称用于输出文件
        actual_model_name = extract_model_name_from_path(model_name)
        print(f"模型名称: {actual_model_name}")
        model_config = None
        local_model = None
        local_tokenizer = None
    else:
        print(f"检测到API模型")
        actual_model_name = model_name
        model_config = None
        local_model = None
        local_tokenizer = None
    
    # 2. 加载配置
    try:
        dataset_config = get_dataset_config(dataset_name)
        
        if not is_local:
            model_config = get_model_config(model_name)
        
        # 使用实际的模型名称生成输出路径
        output_path = get_output_path(dataset_name, actual_model_name, 0, input_key)
        
        if os.path.exists(output_path):
            print(f"结果文件已存在: {output_path}")
            return

    except (ValueError, FileNotFoundError) as e:
        print(f"配置错误: {e}")
        return
    
    print(f"数据库类型: {dataset_config['db_type']}")
    print(f"对象类型: {dataset_config['object_type']}")
    print(f"数据集路径: {dataset_config['path']}")
    print(f"结果保存路径: {output_path}")
    
    if is_local:
        print(f"\n本地模型配置:")
        print(f"  模型路径: {model_name}")
        print(f"  模型名称: {actual_model_name}")
        print(f"  max_new_tokens: 动态估算 (基于gold_plsql词数)")
    else:
        print(f"\n模型配置:")
        print(f"  API类型: {model_config.get('type', 'N/A')}")
        print(f"  模型名称: {model_config.get('model', 'N/A')}")
        print(f"\nLLM调用配置:")
        print(f"  最大tokens: {max_tokens}")
        print(f"  温度: {LLM_CALL_CONFIG.get('temperature', 'N/A')}")
        
        # 创建带有自定义 max_tokens 的配置副本
        llm_call_config = LLM_CALL_CONFIG.copy()
        llm_call_config['max_tokens'] = max_tokens
    
    print("=" * 60)
    
    # 3. 如果是本地模型，加载模型
    if is_local:
        try:
            local_model, local_tokenizer = load_local_model(model_name)
        except Exception as e:
            print(f"加载本地模型失败: {e}")
            return
    
    # 4. 加载数据集
    print(f"\n加载数据集...")
    seeds = load_dataset(dataset_config['path'])
    if not seeds:
        print("数据集为空或加载失败。")
        return
    
    print(f"数据集大小: {len(seeds)} 条")
    
    # 5. 初始化数据集路径配置
    print("初始化数据集路径配置...")
    oracle_util.initialize_dataset_paths(dataset_name)
    
    # 6. 加载数据库 Schema
    print("加载数据库 Schema...")
    oracle_db_schema_dict = oracle_util.get_database_schema_json()
    
    results = []
    error_count = 0

    # 7. 处理每条数据
    print("\n开始处理数据...")
    print("=" * 60)
    
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] 处理中...")
        
        # 提取必要字段（将简写映射到实际字段名）
        input_key_map = {"ir": "ir", "sum": "summary", "nl": "natural_language", "gr": "generated_ir"}
        actual_key = input_key_map.get(input_key, input_key)
        ir_text = seed.get(actual_key, "")
        database_name = seed.get("database_name", "")
        tables = seed.get("tables", [])
        gold_plsql = seed.get("plsql", "")
        
        # 检查必要字段
        if not ir_text or not database_name or not tables:
            print(f"  跳过: 缺少必要字段 ({input_key}/database_name/tables)")
            error_count += 1
            # 即使出错也要将item加入results，保证数量对齐
            result_item = seed.copy()
            result_item["predicted_plsql"] = f"Error: missing required fields ({input_key}/database_name/tables)"
            results.append(result_item)
            continue
        
        # 构建 Schema 信息
        try:
            detailed_database_schema = oracle_db_schema_dict[database_name]
            table_schemas_str = oracle_util.generate_schema_prompt_from_dict(
                detailed_database_schema, tables
            )
            candidate_tables_str = ", ".join(sorted(tables))
            databases_schema_info = f"Tables: {candidate_tables_str}\nDetails:\n{table_schemas_str}"
        except KeyError as e:
            print(f"  跳过: 数据库或表信息缺失 ({e})")
            error_count += 1
            # 即使出错也要将item加入results，保证数量对齐
            result_item = seed.copy()
            result_item["predicted_plsql"] = "Error: database or table information is missing"
            results.append(result_item)
            continue

        # 调用模型进行推理
        try:
            # 构建 Prompt (消息格式，用于统一输出)
            prompt_messages = ORACLE_ZERO_SHOT_PROMPT.format_messages(
                databases_schema_info=databases_schema_info,
                ir_text=ir_text
            )
            
            if is_local:
                # 本地模型推理
                # 构建 Prompt (字符串格式，用于实际推理)
                prompt_text = format_prompt_for_local_model(
                    databases_schema_info=databases_schema_info,
                    ir_text=ir_text
                )
                
                # 动态计算max_new_tokens（使用tokenizer精确计算）
                current_max_new_tokens = estimate_max_new_tokens(
                    gold_plsql=gold_plsql,
                    buffer_ratio=1.5,
                    min_tokens=256,
                    max_tokens=1596,
                    tokenizer=local_tokenizer
                )
                print(f"  max_new_tokens: {current_max_new_tokens} (gold_plsql token数: {len(local_tokenizer.encode(gold_plsql, add_special_tokens=False)) if gold_plsql else 0})")
                
                # 使用本地模型生成
                generated_texts = generate_with_local_model(
                    model=local_model,
                    tokenizer=local_tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=current_max_new_tokens
                )
                
                # 取第一个生成结果
                raw_content = generated_texts[0] if generated_texts else ""
                
            else:
                # API模型推理
                # 调用 LLM
                def llm_call(llm):
                    return llm.invoke(prompt_messages)
                
                response = call_llm_with_retry(
                    llm_func_factory=llm_call,
                    model_config=model_config,
                    llm_call_config=llm_call_config,
                    max_retries=3,
                    timeout=120.0,
                    verbose=False  # 关闭详细日志
                )
                
                raw_content = response.content
            
            # 提取预测代码
            predicted_plsql = extract_plsql_from_tags(raw_content)
            
            # 输出完整Prompt（统一为消息格式）
            print(f"\n  ===== 完整Prompt =====")
            for msg in prompt_messages:
                print(f"  角色: {msg.type}")
                print(f"  内容:\n{msg.content}")
                print(f"  {'-' * 40}")

            print(f"\n  =============== 完整输出 ================")
            print(raw_content)
            
            print(f"\n  ===== 预测的PL/SQL代码 =====")
            print(predicted_plsql)
            
            result_item = seed.copy()
            result_item["predicted_plsql"] = raw_content
            results.append(result_item)

        except Exception as e:
            print(f"  错误: {str(e)}")
            import traceback
            traceback.print_exc()
            error_count += 1
            # 即使出错也要将item加入results，保证数量对齐
            result_item = seed.copy()
            result_item["predicted_plsql"] = "Error: model inference failure or PL/SQL extraction failure"
            results.append(result_item)

    # 8. 保存结果
    print("\n" + "=" * 60)
    print("保存结果...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 9. 输出统计信息
    total = len(results)
    
    print("=" * 60)
    print("实验完成!")
    print("=" * 60)
    print(f"结果文件: {output_path}")
    print(f"总计: {len(seeds)} 条")
    print(f"成功处理: {total} 条")
    print(f"错误/跳过: {error_count} 条")
    print("=" * 60)

    # 显式释放 GPU 资源，防止下一个模型加载时卡住
    if is_local and local_model is not None:
        print("正在释放 GPU 资源...")
        del local_model
        del local_tokenizer
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("GPU 资源已释放")
        except Exception as e:
            print(f"释放 GPU 资源时出错: {e}")

    return results


def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(
        description="Oracle Zero-Shot NL2PL/SQL 实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用API模型
  python oracle_nl2plsql_zero_shot.py --dataset oracle_spider_function_test --model gpt-4o
  
  # 使用本地模型
  python oracle_nl2plsql_zero_shot.py --dataset oracle_spider_function_test --model /home/zhanghang/opt/projects/researchprojects/text2PLSQL/models/starcoder2-3b
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称 (如: oracle_spider_function_test)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称 (如: gpt-4o, deepseek, gemini, claude-4, glm-4) 或本地模型路径 (如: /path/to/starcoder2-3b)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="最大token数 (默认: 8192)"
    )
    
    parser.add_argument(
        "--input-key",
        type=str,
        choices=["ir", "sum", "nl", "gr"],
        default="ir",
        help="输入文本的字段名 (ir/sum/nl/gr, 默认: ir)"
    )
    
    args = parser.parse_args()
    
    run_oracle_zero_shot_experiment(
        args.dataset, 
        args.model, 
        max_tokens=args.max_tokens,
        input_key=args.input_key
    )


if __name__ == "__main__":
    main()
