import os
import json
import sys
import argparse
import re

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(experiments_dir)
sys.path.append(project_root)

from experiments.utils import postgres_util, plsql_util
from experiments.settings.expr_setting import RESULTS_DIR

def extract_info_from_filename(filename):
    """
    从文件名中提取数据集名称、数据库名称和类型
    
    文件名格式示例:
    - postgres_spider_function_test-deepseek-0_shot
    - postgres_omni_procedure_test-gpt4-3_shot
    - postgres_plforge_trigger_test-claude
    
    Args:
        filename: 文件名（不含 .json 后缀）
    
    Returns:
        tuple: (数据集名称, 数据库名称, 类型)
            - 数据集名称: postgres_spider_function_test (用于 initialize_dataset_paths)
            - 数据库名称: spider, omni, plforge
            - 类型: function, procedure, trigger
    """
    # 先按 '-' 分割，获取数据集名称部分
    dataset_name = filename.split('-')[0]
    
    # 按下划线分割数据集名称
    parts = dataset_name.split('_')
    
    # 格式: 数据库类型_数据库名称_类型_test
    # 例如: postgres_spider_function_test
    if len(parts) >= 4:
        db_type = parts[0].lower()  # postgres 或 oracle
        db_name = parts[1].lower()  # spider, omni, plforge
        plsql_type = parts[2].lower()  # function, procedure, trigger
        
        # 验证数据库类型
        if db_type not in ['postgres', 'oracle']:
            raise ValueError(f"无法从文件名 '{filename}' 中识别数据库类型。应为 'postgres' 或 'oracle'")
        
        # 验证数据库名称
        if db_name not in ['spider', 'omni', 'plforge']:
            raise ValueError(f"无法从文件名 '{filename}' 中识别数据库名称。文件名应包含 'spider'、'omni' 或 'plforge'")
        
        # 验证类型
        if plsql_type not in ['function', 'procedure', 'trigger']:
            raise ValueError(f"无法从文件名 '{filename}' 中识别类型。文件名应包含 'function'、'procedure' 或 'trigger'")
        
        return dataset_name, db_name, plsql_type
    
    # 无法识别，抛出错误
    raise ValueError(f"无法从文件名 '{filename}' 中识别信息。文件名格式应为: 数据库类型_数据库名称_类型_test-...")

def evaluate_postgres_json(file_path, host=None, port=None):
    """
    读取JSON文件并评测 PostgreSQL PL/pgSQL 代码的准确性。
    
    Args:
        file_path: JSON文件路径
        host: 数据库主机地址
        port: 数据库端口
    """
    print(f"文件路径: {file_path}")
    
    # 设置数据库连接信息
    if host or port:
        postgres_util.set_db_connection_info(host, port)
    
    # 从文件名中提取数据集名称、数据库名称和类型
    filename = os.path.basename(file_path).replace('.json', '')
    dataset_name, db_name, plsql_type = extract_info_from_filename(filename)
    print(f"数据集名称: {dataset_name}")
    print(f"数据库名称: {db_name}")
    print(f"类型: {plsql_type}")
    
    # 初始化数据集路径（使用数据集名称）
    postgres_util.initialize_dataset_paths(dataset_name)
    
    # 加载 schema_dict 用于给标识符加双引号
    schema_dict = postgres_util.get_database_schema_json()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_count = len(data)
    em_score = 0  # Exact Match score
    ex_score = 0  # Execution score
    ex_error_count = 0  # EX 执行出错的数量
    
    print(f"正在评测 Postgres 数据，共 {total_count} 条记录...")
    print("=" * 60)

    for idx, item in enumerate(data, 1):
        gold_plsql = item["plsql"]
        raw_predicted_plsql = plsql_util.extract_plsql_from_tags(item["predicted_plsql"])
        
        database_name = item["database_name"]
        call_sqls = item["call_sqls"]
        
        if db_name == "spider":
            # 只给 predicted_plsql 中的表名和列名加双引号
            quoted_predicted_plsql = postgres_util.quote_plsql_identifiers(raw_predicted_plsql, database_name, schema_dict)
        else:
            # 其他数据库不进行标识符加引号处理
            quoted_predicted_plsql = raw_predicted_plsql
        
        # 后处理
        processed_gold_plsql = plsql_util.post_process_plsql(gold_plsql)
        processed_predicted_plsql = plsql_util.post_process_plsql(quoted_predicted_plsql)
        
        print(f"\n[{idx}/{total_count}] 评测数据库: {database_name}")
        print(f"  database_name: {database_name}")
        print(f"  [原始] gold_plsql: {gold_plsql}")
        print(f"  [原始] predicted_plsql: {raw_predicted_plsql}")
        print(f"  [处理后] gold_plsql: {processed_gold_plsql}")
        print(f"  [处理后] predicted_plsql: {processed_predicted_plsql}")
        print(f"  call_sqls: {call_sqls}")
        
        # EM 评测：使用 AST 语义匹配
        is_em_match = postgres_util.is_exact_match(processed_gold_plsql, processed_predicted_plsql)
        if is_em_match:
            em_score += 1
            print(f"  EM: ✓ (语义匹配)")
        else:
            print(f"  EM: ✗ (语义不匹配)")
        
        # EX 评测：根据从文件名提取的类型使用相应的比较函数
        try:
            # 根据类型选择比较方法
            if plsql_type == 'function':
                # 对于 function，使用 compare_plsql_function
                is_ex_match = postgres_util.compare_plsql_function(
                    database_name, 
                    gold_plsql, 
                    quoted_predicted_plsql, 
                    call_sqls
                )
            else:
                # 对于 procedure 和 trigger，使用 compare_plsql
                is_ex_match = postgres_util.compare_plsql(
                    database_name, 
                    gold_plsql, 
                    quoted_predicted_plsql, 
                    call_sqls,
                    include_system_tables=False
                )
            
            if is_ex_match:
                ex_score += 1
                print(f"  EX: ✓ (执行结果一致)")
            else:
                print(f"  EX: ✗ (执行结果不一致)")
            
        except Exception as e:
            ex_error_count += 1
            print(f"  EX: ✗ (执行出错: {str(e)[:100]})")

    print("\n" + "=" * 60)
    print(f"评测完成 (PostgreSQL)")
    print("-" * 60)
    print(f"文件路径: {file_path}")
    print(f"数据集名称: {dataset_name}")
    print(f"数据库名称: {db_name}")
    print(f"类型: {plsql_type}")
    print("-" * 60)
    
    em_accuracy = em_score / total_count if total_count > 0 else 0
    ex_accuracy = ex_score / total_count if total_count > 0 else 0
    
    print(f"EM (Exact Match - 语义匹配):")
    print(f"  匹配数量: {em_score}/{total_count}")
    print(f"  准确率: {em_accuracy:.4f} ({em_accuracy*100:.2f}%)")
    print("-" * 60)
    print(f"EX (Execution - 执行准确率):")
    print(f"  正确数量: {ex_score}/{total_count}")
    print(f"  准确率: {ex_accuracy:.4f} ({ex_accuracy*100:.2f}%)")
    print(f"  执行失败: {ex_error_count} 条")
    print("-" * 60)
    print(f"SUMMARY: File={filename}, EM={em_score} ({em_accuracy*100:.2f}%), EX={ex_score} ({ex_accuracy*100:.2f}%), Total={total_count}, Dataset={dataset_name}, DB={db_name}, Type={plsql_type}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测 Postgres PL/pgSQL 代码准确性")
    parser.add_argument("filename", type=str, help="结果文件名（不含 .json 后缀）")
    parser.add_argument("--host", type=str, help="数据库主机地址", default=None)
    parser.add_argument("--port", type=str, help="数据库端口", default=None)
    
    args = parser.parse_args()
    
    # 自动拼接完整路径
    file_path = os.path.join(RESULTS_DIR, f"{args.filename}.json")
    
    evaluate_postgres_json(file_path, args.host, args.port)