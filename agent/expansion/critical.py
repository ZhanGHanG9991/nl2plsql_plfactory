import json
import re
from langchain.prompts import ChatPromptTemplate

from util.llm_util import init_llm_with_random_provider, call_llm_with_retry

from util.postgres_util import cleanup_plsql_objects as clean_plpgsql, restore_databases, check_plsql_executability as check_postgres_plsql
from util.oracle_util import cleanup_plsql_objects as clean_oracle_plsql, recreate_database_with_context, check_plsql_executability as check_oracle_plsql
from state.expansion_state import ExpansionState
from config.common import pg_config, seed_generation_config, weak_llm_config

from util.postgres_util import generate_schema_prompt_from_dict as generate_pg_schema
from util.postgres_util import restore_databases
from util.oracle_util import generate_schema_prompt_from_dict as generate_oracle_schema

ALLOWED_DATABASE_TYPES = ["postgresql", "mysql", "oracle"]

generate_call_prompt = ChatPromptTemplate([
    (
        "user",
        "You are an expert in the field of databases."
        "Below is the PLSQL code defining a function / procedure / trigger in the {database_type} database. Please select appropriate function parameters based on the data in the relevant table and generate a simple SQL code snippet to call this function / procedure / trigger.\n"
        "You should generate 1 sql to call the function / procedure / trigger."
        "### PLSQL Code:\n {plsql_code} \n"
        "### Database Tables:\n {tables} \n"
        "### Output Format:\n"
        "IMPORTANT: Output ONLY the SQL code snippet to call the function / procedure / trigger, WITHOUT any additional explanations, descriptions, or extra text.\n"
        "Each generated query must be wrapped in <start-plsql> and <end-plsql> tags:\n\n"
        "<start-plsql>\n"
        "[SQL here]\n"
        "<end-plsql>\n\n"
        "Do NOT include any text before or after the queries. Output ONLY the queries in the specified format."
        "### Example1:\n"
        "<start-plsql>\n"
        "select get_employee_count(10);\n"
        "<end-plsql>\n"
        "### Example2: (Oracle procedure)\n"
        "<start-plsql>\n"
        "BEGIN\n  sp('now', 'Pinot Noir', 32, 'California');\n  commit;\nEND;\n"
        "<end-plsql>\n"
    )
])

def _call_critical_llm_with_retry(prompt, max_retries: int = 3, timeout: float = 120.0):
    """
    使用超时和重试机制调用critical模型LLM
    
    Args:
        prompt: 要发送给LLM的prompt
        max_retries: 最大重试次数，默认3次
        timeout: 每次调用的超时时间（秒），默认120秒
    
    Returns:
        LLM响应对象
    """
    critical_model_cfg = weak_llm_config.get("critical_model", {})
    
    def llm_call(llm):
        """LLM调用函数"""
        return llm.invoke(prompt)
    
    # 使用超时重试机制调用
    response = call_llm_with_retry(
        llm_func_factory=llm_call,
        model_config=critical_model_cfg,
        max_retries=max_retries,
        timeout=timeout,
        model_name="critical_model",
        verbose=True
    )
    
    return response


def generate_call_sql(state: ExpansionState, plsql_code):
    dialect = state["dialect"]
    if dialect == "postgresql":
        database_schema_str = generate_pg_schema(state["selected_detailed_database_schema"], state["selected_tables"])
    elif dialect == "oracle":
        database_schema_str = generate_oracle_schema(state["selected_detailed_database_schema"], state["selected_tables"])
    
    prompt = generate_call_prompt.format_messages(
        database_type=state["dialect"],
        plsql_code=plsql_code,
        tables=database_schema_str
    )
    response = _call_critical_llm_with_retry(prompt, max_retries=3, timeout=120.0)
    response_content = response.content.strip()
    
    print("[INFO] LLM Generate Call Response:")
    print(response_content)

    pattern = r'<start-plsql>\s*(.*?)\s*<end-plsql>'
    queries = re.findall(pattern, response_content, re.DOTALL | re.IGNORECASE)
    
    queries = [query.strip() for query in queries if query.strip()]
    
    if len(queries) == 0:
        pattern = r'<start-plsql>\s*(.*?)\s*</end-plsql>'
        queries = re.findall(pattern, response_content, re.DOTALL | re.IGNORECASE)
        queries = [query.strip() for query in queries if query.strip()]
    
    if len(queries) == 0:
        pattern = r'<start-plsql>\s*(.*?)\s*</start-plsql>'
        queries = re.findall(pattern, response_content, re.DOTALL | re.IGNORECASE)
        queries = [query.strip() for query in queries if query.strip()]

    if dialect == "oracle":
        for i in range(len(queries)):
            query = queries[i].strip()
            if query.endswith(";") and (not query.endswith("END;")):
                queries[i] = query.rstrip(";")
    
    return queries

def save_to_json(file_path, entries):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    except FileNotFoundError:
        data = []
    
    if not isinstance(data, list):
        data = [data]
    data.extend(entries)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_garbage(dialect, entries):
    if dialect not in ALLOWED_DATABASE_TYPES:
        raise Exception(f"Dialect {dialect} is not supported")
    if dialect == "postgresql":
        save_to_json(seed_generation_config["pg_generation_garbage_path"], entries)
    elif dialect == "oracle":
        save_to_json(seed_generation_config["oc_generation_garbage_path"], entries)


def critical(state: ExpansionState):
    dialect = state["dialect"]

    if dialect not in ALLOWED_DATABASE_TYPES:
        raise Exception(f"Dialect {dialect} is not supported")

    state["critical_epoch"] += 1
    critical_epoch = state["critical_epoch"]

    if critical_epoch == 5:
        return state

    if 2 <= critical_epoch <= 3:
        if not any(state['need_correction']):
            state["ir_plsqls"] = []
            for i, item in enumerate(state["expansion_plsqls"]):
                plsql = item["plsql"]
                state["ir_plsqls"].append({"plsql": plsql, 
                                           "ir": item["ir"],
                                           "database_name": state["selected_database_name"], 
                                           "tables": state["selected_tables"]})
            state["critical_epoch"] = 4
            return state
        
    database_name = state["selected_database_name"]

    execution_infos = []
    need_correction = []

    for i, item in enumerate(state["expansion_plsqls"]):
        plsql_code = item["plsql"]
        execution_result = None
        if critical_epoch == 1 or state['need_correction'][i]:
            call_sqls = generate_call_sql(state, plsql_code)
            print("[INFO] Generated call sql:")
            print(call_sqls)
            if dialect == "postgresql":
                execution_result = check_postgres_plsql(plsql_code, call_sqls, database_name)
                if clean_plpgsql(plsql_code, database_name) is not None:
                    pg_conn_info = f"host={pg_config['host']} user={pg_config['user']} password={pg_config['password']} dbname={database_name} port={pg_config['port']}"
                    restore_databases(pg_conn_info, pg_config['host'], pg_config['port'], pg_config['user'], pg_config['password'], [database_name])

            elif dialect == "oracle":
                execution_result = check_oracle_plsql(plsql_code, call_sqls, database_name)
                if clean_oracle_plsql(plsql_code, database_name) is not None:
                    recreate_database_with_context(database_name)
        if execution_result is None:
            need_correction.append(False)
            execution_infos.append("")
        else:
            need_correction.append(True)
            execution_infos.append(execution_result)
        
    state['need_correction'] = need_correction
    state['execution_info'] = execution_infos

    if critical_epoch != 4:
        return state
        
    # garbage collect and save ir_plsqls
    state["ir_plsqls"] = []
    garbages = []  
    for i, item in enumerate(state["expansion_plsqls"]):
        plsql = item["plsql"]
        if need_correction[i]:
            garbages.append({"plsql": plsql, "database_name": database_name, "execution_info": execution_infos[i], "tables": state["selected_tables"]})
        else:
            state["ir_plsqls"].append({"plsql": plsql, 
                                       "ir": item["ir"],
                                       "database_name": database_name, 
                                       "tables": state["selected_tables"]})
    
    save_garbage(dialect, garbages)
    return state