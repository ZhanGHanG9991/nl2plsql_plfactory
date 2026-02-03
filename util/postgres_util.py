import os
import psycopg
from psycopg import sql as psql
from psycopg import errors
import subprocess
import pandas as pd
import sqlparse
from sqlparse import sql, tokens
from typing import List, Optional
import re
import json

from tqdm import tqdm
from config.common import pg_config

# Connection info
conn_info = f"host={pg_config['host']} user={pg_config['user']} password={pg_config['password']} dbname={pg_config['dbname']} port={pg_config['port']}"
host = pg_config['host']
port = pg_config['port']
user = pg_config['user']
password = pg_config['password']
input_path = pg_config["input_path"]
db_schema_graph_path = pg_config["db_schema_graph_path"]
db_schema_dict_path = pg_config["db_schema_dict_path"]

def get_tables_info(database_name):
    conn_db_info = f"""host={host} dbname={database_name} user={user} password={password}"""
    tables_info = {}
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT tablename
                            FROM pg_catalog.pg_tables
                            WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';""")
            result = cur.fetchall()
            table_names = [table_name[0] for table_name in result]

            for table_name in table_names:
                cur.execute(f"""SELECT column_name, data_type
                                FROM information_schema.columns
                                WHERE table_name = '{table_name}';""")
                result = cur.fetchall()
                tables_info[table_name] = [item for item in result]
                
    return tables_info

def get_database_schema(database_name):
    """获取数据库schema信息，返回符合DatabaseSchema类型的字典
    
    Returns:
        Dict包含:
        - table_names: List[str] - 所有表名列表
        - tables: Dict[str, List[str]] - 表名到列名列表的映射
    """
    conn_db_info = f"""host={host} dbname={database_name} user={user} password={password}"""
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            # 获取所有表名
            cur.execute(f"""SELECT tablename
                            FROM pg_catalog.pg_tables
                            WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';""")
            result = cur.fetchall()
            table_names = [table_name[0] for table_name in result]
            
            # 获取每个表的列名
            tables = {}
            for table_name in table_names:
                cur.execute(f"""SELECT column_name
                                FROM information_schema.columns
                                WHERE table_name = '{table_name}'
                                ORDER BY ordinal_position;""")
                result = cur.fetchall()
                tables[table_name] = [column_name[0] for column_name in result]
    
    return {
        'table_names': table_names,
        'tables': tables
    }

def get_detailed_database_schema(database_name, sample_limit=3):
    """获取详细的数据库schema信息，适合text2sql任务
    
    Args:
        database_name: 数据库名
        sample_limit: 每表采样数据行数
    
    Returns:
        dict: 包含详细schema信息的字典，结构如下：
            {
                'database_name': str,
                'tables': {
                    'table1': {
                        'columns': [
                            {
                                'name': str,
                                'data_type': str,
                                'is_nullable': str,
                                'constraint_type': str,
                                'comment': str,
                                'examples': list
                            }
                        ],
                        'sample_data': list,
                        'column_names': list
                    }
                },
                'relationships': list,
                'formatted_string': str  # 保持原有格式的字符串
            }
    """
    conn_db_info = f"host={host} dbname={database_name} user={user} password={password}"
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            # 获取所有表名
            cur.execute("""
                SELECT tablename 
                FROM pg_catalog.pg_tables 
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY tablename;
            """)
            table_names = [row[0] for row in cur.fetchall()]
            
            schema_dict = {
                'database_name': database_name,
                'tables': {},
                'relationships': []
            }
            
            for table_name in table_names:                
                # 获取列详细信息
                cur.execute("""
                    SELECT 
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        CASE WHEN tc.constraint_type = 'PRIMARY KEY' THEN 'PRIMARY KEY' ELSE NULL END as primary_key,
                        col_description(pgc.oid, c.ordinal_position) as column_comment
                    FROM information_schema.columns c
                    LEFT JOIN information_schema.key_column_usage kcu
                        ON c.table_name = kcu.table_name 
                        AND c.column_name = kcu.column_name
                    LEFT JOIN information_schema.table_constraints tc
                        ON kcu.table_name = tc.table_name 
                        AND kcu.constraint_name = tc.constraint_name
                        AND tc.constraint_type = 'PRIMARY KEY'
                    LEFT JOIN pg_class pgc ON pgc.relname = c.table_name
                    WHERE c.table_name = %s
                    ORDER BY c.ordinal_position;
                """, (table_name,))
                
                columns_info = cur.fetchall()
                
                # 数据采样
                cur.execute(f"""
                    SELECT * FROM "{table_name}" LIMIT {sample_limit};
                """)
                sample_data = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]
                
                # 为每个列收集样例数据
                column_examples = {}
                if sample_data:
                    for i, row in enumerate(sample_data):
                        for j, col_name in enumerate(column_names):
                            if j < len(row):
                                if col_name not in column_examples:
                                    column_examples[col_name] = []
                                value_example = str(row[j])
                                if len(value_example) > 30:
                                    value_example = value_example[:30] + "..."
                                else:
                                    value_example = row[j]
                                if value_example not in column_examples[col_name]:
                                    column_examples[col_name].append(value_example)
                
                # 构建表结构字典
                table_info = {
                    'columns': [],
                    'sample_data': sample_data,
                    'column_names': column_names
                }
                
                for col in columns_info:
                    column_name, data_type, is_nullable, constraint_type, col_comment = col
                    
                    # 添加到字典
                    column_dict = {
                        'name': column_name,
                        'data_type': data_type,
                        'is_nullable': is_nullable,
                        'constraint_type': constraint_type,
                        'comment': col_comment,
                        'examples': column_examples.get(column_name, [])[:5]  # 最多5个样例
                    }
                    table_info['columns'].append(column_dict)
                    
                # 保存表信息到字典
                schema_dict['tables'][table_name] = table_info
              
            # 外键关系部分
            all_relationships = []
            
            for table_name in table_names:
                cur.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
                """, (table_name,))
                
                relationships = cur.fetchall()
                if relationships:
                    for rel in relationships:
                        col_name, foreign_table, foreign_col = rel
                        relationship_str = f"{table_name}.{col_name} = {foreign_table}.{foreign_col}"
                        all_relationships.append(relationship_str)
                        
                        # 添加到字典
                        schema_dict['relationships'].append({
                            'source_table': table_name,
                            'source_column': col_name,
                            'target_table': foreign_table,
                            'target_column': foreign_col
                        })
                        
    return schema_dict

# 辅助函数：从字典重新生成特定表的prompt
def generate_schema_prompt_from_dict(schema_dict, table_names=None):
    """从schema字典生成特定表的prompt字符串
    
    Args:
        schema_dict: get_detailed_database_schema返回的字典
        table_names: 要包含的表名列表，如果为None则包含所有表
    
    Returns:
        str: 格式化的schema描述字符串
    """
    if table_names is None:
        table_names = list(schema_dict['tables'].keys())
    
    formatted_parts = []
    
    for table_name in sorted(table_names):
        if table_name not in schema_dict['tables']:
            continue
            
        table_info = schema_dict['tables'][table_name]
        
        table_schema = f"Table: {table_name}\n"
        table_schema += "Columns:\n"
        
        for col in table_info['columns']:
            col_info = f"  - {col['name']} ({col['data_type']})"
            
            if col['constraint_type'] == 'PRIMARY KEY':
                col_info += " PRIMARY KEY"
            
            if col['is_nullable'] == 'NO':
                col_info += " NOT NULL"
            
            if col['examples']:
                col_info += f" examples: {col['examples']}"
            
            if col['comment']:
                col_info += f" - {col['comment']}"
            
            table_schema += col_info + "\n"
        
        formatted_parts.append(table_schema)
    
    # 添加关系信息
    relationship_summary = "\nTable Relationships:\n"
    if schema_dict['relationships']:
        relationships = []
        for rel in schema_dict['relationships']:
            if rel['source_table'] in table_names and rel['target_table'] in table_names:
                rel_str = f"{rel['source_table']}.{rel['source_column']} = {rel['target_table']}.{rel['target_column']}"
                relationships.append(rel_str)
        
        if relationships:
            relationship_summary += "\n".join(sorted(list(set(relationships))))
        else:
            relationship_summary += "No relationships between selected tables."
    else:
        relationship_summary += "No foreign key relationships found."
    
    formatted_parts.append(relationship_summary)
    
    return "\n".join(formatted_parts) + "\n"

def _get_foreign_key_relations(database_name):
    """获取数据库中所有表的外键关系
    
    Args:
        database_name: 数据库名称
        
    Returns:
        Dict[str, List[str]]: 字典，键为表名，值为与该表有外键关联的其他表名列表
    """
    conn_db_info = f"""host={host} dbname={database_name} user={user} password={password}"""
    foreign_key_relations = {}
    
    try:
        with psycopg.connect(conn_db_info) as conn:
            with conn.cursor() as cur:
                # 获取所有用户表
                cur.execute("""
                    SELECT tablename
                    FROM pg_catalog.pg_tables
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY tablename;
                """)
                tables = [row[0] for row in cur.fetchall()]
                
                # 初始化所有表的外键关系列表
                for table in tables:
                    foreign_key_relations[table] = []
                
                # 查询所有外键关系
                cur.execute("""
                    SELECT
                        tc.table_name AS from_table,
                        ccu.table_name AS to_table
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY tc.table_name;
                """)
                
                for from_table, to_table in cur.fetchall():
                    # 添加双向关系
                    if to_table not in foreign_key_relations[from_table]:
                        foreign_key_relations[from_table].append(to_table)
                    if from_table not in foreign_key_relations[to_table]:
                        foreign_key_relations[to_table].append(from_table)
                
    except Exception as e:
        print(f"Error getting foreign key relations for database {database_name}: {e}")
        return {}
    
    return foreign_key_relations

def get_database_schema_graph():
    """获取所有PostgreSQL数据库的schema图谱
    
    Returns:
        Dict: 格式为 {
            "数据库名1": {
                "tables": ["table1", "table2"],
                "table1": [与table1有外键相连的table名]
            },
            ...
        }
    """
    if os.path.exists(db_schema_graph_path):
        # 如果文件已存在，直接加载返回
        with open(db_schema_graph_path, 'r', encoding='utf-8') as f:
            db_schema_graph = json.load(f)
        return db_schema_graph
    
    # 在线构建Dict
    db_schema_graph = {}
    
    # 获取input_path目录下的所有.sql文件
    if not os.path.exists(input_path):
        print(f"Warning: input_path {input_path} does not exist")
        return db_schema_graph
    
    sql_files = [f for f in os.listdir(input_path) if f.endswith('.sql')]
    
    for sql_file in sql_files[:10]:
        # 从文件名中提取数据库名（去除.sql后缀）
        database_name = sql_file[:-4]
        
        try:
            print(f"Processing database: {database_name}")
            
            # 获取所有用户表
            all_tables = get_all_user_tables(database_name)
            
            # 获取外键关系
            foreign_key_relations = _get_foreign_key_relations(database_name)
            
            # 构建该数据库的schema图
            db_schema_graph[database_name] = {
                "tables": all_tables
            }
            
            # 添加每个表的外键关系
            for table in all_tables:
                db_schema_graph[database_name][table] = foreign_key_relations.get(table, [])
                
        except Exception as e:
            print(f"Error processing database {database_name}: {e}")
            continue
    
    # 保存至db_schema_graph_path
    try:
        os.makedirs(os.path.dirname(db_schema_graph_path), exist_ok=True)
        with open(db_schema_graph_path, 'w', encoding='utf-8') as f:
            json.dump(db_schema_graph, f, ensure_ascii=False, indent=2)
        print(f"Database schema graph saved to {db_schema_graph_path}")
    except Exception as e:
        print(f"Error saving database schema graph: {e}")
    
    return db_schema_graph


def get_database_schema_json():
    """获取所有 PostgreSQL 数据库的 schema JSON

    """
    if os.path.exists(db_schema_dict_path):
        # 如果文件已存在，直接加载返回
        with open(db_schema_dict_path, 'r', encoding='utf-8') as f:
            db_schema_dict = json.load(f)
        return db_schema_dict
    
    # 在线构建Dict
    db_schema_dict = {}
    
    # 获取input_path目录下的所有.sql文件
    if not os.path.exists(input_path):
        print(f"Warning: input_path {input_path} does not exist")
        return db_schema_dict
    
    sql_files = [f for f in os.listdir(input_path) if f.endswith('.sql')]
    
    for sql_file in tqdm(sql_files):
        # 从文件名中提取数据库名（去除.sql后缀）
        database_name = sql_file[:-4]
        
        try:
            print(f"Processing database: {database_name}")
            db_schema_dict[database_name] = get_detailed_database_schema(database_name)
                
        except Exception as e:
            print(f"Error processing database {database_name}: {e}")
            continue
    
    # 保存至 db_schema_dict
    try:
        os.makedirs(os.path.dirname(db_schema_dict), exist_ok=True)
        with open(db_schema_dict_path, 'w', encoding='utf-8') as f:
            json.dump(db_schema_dict, f, ensure_ascii=False, indent=2)
        print(f"Database schema dict saved to {db_schema_dict}")
    except Exception as e:
        print(f"Error saving database schema dict: {e}")
    
    return db_schema_dict

def get_all_user_tables(database_name):
    """获取数据库中所有用户表的名称"""
    conn_db_info = f"""host={host} dbname={database_name} user={user} password={password}"""
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            # 获取所有用户表（排除系统表）
            cur.execute("""
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY tablename;
            """)
            result = cur.fetchall()
            return [table_name[0] for table_name in result]

def get_important_system_tables():
    """Return a list of system tables that are important for PostgreSQL"""
    return [
        'pg_indexes',               # 索引信息
        'pg_constraints',           # 约束信息
        'pg_triggers',              # 触发器信息
        'pg_sequences',             # 序列信息
        'pg_views',                 # 视图信息
        'pg_user_mappings',         # 用户映射
        'pg_policies',              # 行级安全策略
        'pg_rules'                  # 规则信息
    ]

def fetch_system_table_data(database_conn_info, system_table):
    """获取系统表数据，处理可能的权限或存在性问题"""
    try:
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                # 检查表是否存在
                cur.execute(f"""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM information_schema.tables 
                        WHERE table_name = '{system_table}'
                        AND table_schema IN ('pg_catalog', 'information_schema')
                    ) OR EXISTS (
                        SELECT 1 
                        FROM information_schema.views 
                        WHERE table_name = '{system_table}'
                        AND table_schema IN ('pg_catalog', 'information_schema')
                    );
                """)
                
                if not cur.fetchone()[0]:
                    return None
                
                # 尝试查询系统表数据
                cur.execute(f"SELECT * FROM {system_table} ORDER BY 1;")
                result = cur.fetchall()
                return result
    except Exception as e:
        print(f"Warning: Could not fetch data from system table {system_table}: {e}")
        return None

# 删除再创建数据库的函数
def recreate_databases(conn_info, databases, maintenance_db="postgres"):
    dsn = psycopg.conninfo.make_conninfo(conn_info, dbname=maintenance_db)

    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for db_name in databases:
                if db_name == maintenance_db:
                    continue

                # 断开目标数据库的所有连接
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                    """,
                    (db_name,),
                )

                ident = psql.Identifier(db_name)

                cur.execute(psql.SQL("DROP DATABASE IF EXISTS {}").format(ident))
                cur.execute(psql.SQL("CREATE DATABASE {}").format(ident))

def import_database(host, port, user, password, dbname, input_file):
    # 加上 -v ON_ERROR_STOP=1，遇到第一个 SQL 错误就停止并返回非零状态码
    command = f"psql -h {host} -p {port} -U {user} -d {dbname} -v ON_ERROR_STOP=1 -f {input_file}"
    env = {**os.environ, "PGPASSWORD": password}
    
    print(f"Executing: {command}")
    
    process = subprocess.run(
        command, 
        env=env, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True 
    )
    
    if process.returncode != 0:
        error_msg = f"PSQL Import Failed for {dbname}!\nReturncode: {process.returncode}\nStderr:\n{process.stderr}\nStdout:\n{process.stdout}"
        print(error_msg) 
        raise RuntimeError(error_msg)
    else:
        print(f"Successfully imported {dbname}.")
        # 即使成功，也可以打印 stderr，因为可能有 NOTICE/WARNING
        if process.stderr:
            print(f"Warnings/Notices:\n{process.stderr}")

def restore_databases(conn_info, host, port, user, password, database_names):
    try:
        # 删除并重建数据库
        recreate_databases(conn_info, database_names)

        # 导入数据
        for dbname in database_names:
            input_file = os.path.join(input_path, f"{dbname}.sql")
            import_database(host, port, user, password, dbname.lower(), input_file)
    except Exception as e:
        print(f"Error restoring databases {database_names}: {e}")

def cleanup_plsql_objects(plsql_code, database_name):
    """清理PL/SQL代码中可能创建的函数、存储过程和触发器"""
    try:
        database_conn_info = f"host={host} user={user} password={password} dbname={database_name} port={port}"
        
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                # 分析PL/SQL代码，识别创建的对象
                objects_to_drop = analyze_plsql_objects(plsql_code)
                
                # 清理识别到的对象
                for obj_type, obj_name in objects_to_drop:
                    if obj_type == 'function':
                        cur.execute(f"DROP FUNCTION IF EXISTS {obj_name} CASCADE")
                    elif obj_type == 'procedure':
                        cur.execute(f"DROP PROCEDURE IF EXISTS {obj_name} CASCADE")
                    elif obj_type == 'trigger':
                        cur.execute(f"DROP TRIGGER IF EXISTS {obj_name} CASCADE")
                        
    except Exception as e:
        return str(e)
    return None


def get_plsql_type(plsql_code):
    objects = analyze_plsql_objects(plsql_code)
    for obj_type, _ in objects:
        if obj_type == 'trigger':
            return 'trigger'
    for obj_type, _ in objects:
        return obj_type
    return 'unknown'
    
def analyze_plsql_objects(plsql_code):
    """分析PL/SQL代码，识别创建的对象"""
    objects = []
    patterns = [
        ('function', r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+"?(\w+)"?'),
        ('procedure', r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+"?(\w+)"?'),
        ('trigger', r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+"?(\w+)"?')
    ]
    
    for obj_type, pattern in patterns:
        matches = re.finditer(pattern, plsql_code, re.IGNORECASE)
        objects.extend((obj_type, match.group(1)) for match in matches)
    
    return objects

def execute_sql(database_conn_info, sql):
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            # 设置查询超时
            cur.execute(f"SET statement_timeout = {2 * 1000};")  # timeout单位为毫秒
            cur.execute(sql)

def check_plsql_executability(generated_plsql, call_plsqls, database_name):
    execution_error = None
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name} port={port}"""
        restore_databases(conn_info, host, port, user, password, [database_name])
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                # 设置查询超时
                cur.execute(f"SET statement_timeout = {2 * 1000};")  # timeout单位为毫秒
                cur.execute(generated_plsql)
                for call in call_plsqls:
                    cur.execute(call)
    except errors.Error as e:  # 捕获PostgreSQL特定的错误
        execution_error = str(e.sqlstate) + ":" + str(e)
    except Exception as e:
        execution_error = str(e)
    
    return execution_error

def fetch_query_results(database_conn_info, query):
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            return result


def will_change_data(database_name, plsql_code, call_plsqls, include_system_tables=False):
    """
    判断执行PL/SQL代码和调用列表中的SQL时是否会改变数据库数据
    
    Args:
        database_name: 数据库名称
        plsql_code: 要检查的PL/SQL代码
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否检查系统表的变化
    
    Returns:
        dict: 包含详细变化信息的结果字典
    """
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name} port={port}"""
        
        # 获取所有用户表
        all_user_tables = get_all_user_tables(database_name)
        print(f"Monitoring {len(all_user_tables)} user tables for changes")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Monitoring {len(important_system_tables)} system tables for changes")
        
        # 执行前备份所有表的数据
        before_execution_data = {}
        
        # 备份用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                before_execution_data[table] = pd.DataFrame(result) if result is not None else None
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table} before execution: {e}")
                before_execution_data[table] = None
        
        # 备份系统表数据
        system_tables_before = {}
        for sys_table in important_system_tables:
            system_tables_before[sys_table] = fetch_system_table_data(database_conn_info, sys_table)
        
        # 执行PL/SQL代码和调用语句
        execute_sql(database_conn_info, plsql_code)
        print("Executed PL/SQL code")
        for call in call_plsqls:
            execute_sql(database_conn_info, call)
        
        # 执行后获取所有表的数据
        after_execution_data = {}
        
        # 获取执行后的用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                after_execution_data[table] = pd.DataFrame(result) if result is not None else None
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table} after execution: {e}")
                after_execution_data[table] = None
        
        # 获取执行后的系统表数据
        system_tables_after = {}
        for sys_table in important_system_tables:
            system_tables_after[sys_table] = fetch_system_table_data(database_conn_info, sys_table)
        
        # 比较数据变化
        changed_user_tables = []
        changed_system_tables = []
        
        # 比较用户表数据
        for table in all_user_tables:
            data_before = before_execution_data.get(table)
            data_after = after_execution_data.get(table)
            
            if data_before is None and data_after is None:
                continue
            elif data_before is None or data_after is None:
                changed_user_tables.append(table)
            elif not data_before.equals(data_after):
                changed_user_tables.append(table)
        
        # 比较系统表数据
        if include_system_tables:
            for sys_table in important_system_tables:
                data_before = system_tables_before.get(sys_table)
                data_after = system_tables_after.get(sys_table)
                
                if data_before is None and data_after is None:
                    continue
                elif data_before is None or data_after is None:
                    changed_system_tables.append(sys_table)
                elif data_before != data_after:
                    changed_system_tables.append(sys_table)
        
        # 判断是否有数据变化
        has_data_changes = len(changed_user_tables) > 0 or len(changed_system_tables) > 0
        
        result = {
            'will_change_data': has_data_changes,
            'changed_user_tables': changed_user_tables,
            'changed_system_tables': changed_system_tables,
            'total_user_tables_monitored': len(all_user_tables),
            'total_system_tables_monitored': len(important_system_tables),
            'user_tables_changed_count': len(changed_user_tables),
            'system_tables_changed_count': len(changed_system_tables),
            'changes_detailed': {
                'user_tables': changed_user_tables,
                'system_tables': changed_system_tables
            }
        }
        
        print(f"Data change analysis completed:")
        print(f"Will change data: {has_data_changes}")
        print(f"User tables changed: {len(changed_user_tables)}/{len(all_user_tables)}")
        print(f"System tables changed: {len(changed_system_tables)}/{len(important_system_tables)}")
        
        if changed_user_tables:
            print(f"Changed user tables: {changed_user_tables}")
        if changed_system_tables:
            print(f"Changed system tables: {changed_system_tables}")
        
        return result
    
    except Exception as e:
        print(f"Error in will_change_data: {e}")
        return {
            'will_change_data': None,  # 表示无法确定
            'error': str(e),
            'changed_user_tables': [],
            'changed_system_tables': [],
            'total_user_tables_monitored': 0,
            'total_system_tables_monitored': 0,
            'user_tables_changed_count': 0,
            'system_tables_changed_count': 0
        }


# 简化版本：只返回布尔值
def will_change_data_simple(database_name, plsql_code, call_plsqls, include_system_tables=False):
    """
    简化版本：只返回是否会改变数据的布尔值
    
    Returns:
        bool: True表示会改变数据，False表示不会改变数据，None表示检查出错
    """
    try:
        result = will_change_data(database_name, plsql_code, call_plsqls, include_system_tables)
        return result.get('will_change_data')
    except Exception as e:
        print(f"Error in will_change_data_simple: {e}")
        return None

def compare_plsql(database_name, plsql1, plsql2, call_plsqls, include_system_tables):
    """
    比较两个PL/SQL代码的执行结果
    
    Args:
        database_name: 数据库名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否包含系统表比较
    
    Returns:
        True or False
    """
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name}"""
        
        # 获取所有用户表
        all_user_tables = get_all_user_tables(database_name)
        print(f"Found {len(all_user_tables)} user tables to compare: {all_user_tables}")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Will compare {len(important_system_tables)} system tables")
        
        # 第一次执行
        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql1)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        # 收集第一次执行后的数据
        user_tables_results1 = {}
        system_tables_results1 = {}
        
        # 获取所有用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                user_tables_results1[table] = pd.DataFrame(result)
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table}: {e}")
                user_tables_results1[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results1[sys_table] = fetch_system_table_data(database_conn_info, sys_table)
        
        # 第二次执行
        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql2)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        # 收集第二次执行后的数据
        user_tables_results2 = {}
        system_tables_results2 = {}
        
        # 获取所有用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                user_tables_results2[table] = pd.DataFrame(result)
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table}: {e}")
                user_tables_results2[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results2[sys_table] = fetch_system_table_data(database_conn_info, sys_table)

        # 比较用户表数据
        user_tables_same = True
        user_tables_diff = []
        
        for table in all_user_tables:
            df1 = user_tables_results1.get(table)
            df2 = user_tables_results2.get(table)
            
            if df1 is None and df2 is None:
                continue
            elif df1 is None or df2 is None:
                user_tables_same = False
                user_tables_diff.append(table)
            elif not df1.equals(df2):
                user_tables_same = False
                user_tables_diff.append(table)
        
        # 比较系统表数据
        system_tables_same = True
        system_tables_diff = []
        
        if include_system_tables:
            for sys_table in important_system_tables:
                result1 = system_tables_results1.get(sys_table)
                result2 = system_tables_results2.get(sys_table)
                
                if result1 is None and result2 is None:
                    continue
                elif result1 is None or result2 is None:
                    system_tables_same = False
                    system_tables_diff.append(sys_table)
                elif result1 != result2:
                    system_tables_same = False
                    system_tables_diff.append(sys_table)
        
        # 综合结果
        overall_same = user_tables_same and system_tables_same
        
        result = {
            'overall_same': overall_same,
            'user_tables_same': user_tables_same,
            'system_tables_same': system_tables_same,
            'user_tables_compared': len(all_user_tables),
            'system_tables_compared': len(important_system_tables),
            'user_tables_diff': user_tables_diff,
            'system_tables_diff': system_tables_diff
        }

        print(result)
        
        return result.get('overall_same', False)
    
    except Exception as e:
        print(f"Error in compare_plsql: {e}")
        return False


def compare_plsql_function(database_name, plsql1, plsql2, call_plsqls):
    """
    比较两个PL/SQL函数代码的执行结果
    
    Args:
        database_name: 数据库名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
    
    Returns:
        True or False
    """
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name}"""

        # 第一次执行
        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql1)

        # 收集第一次执行的结果
        function_results1 = {}
        for i, call_plsql in enumerate(call_plsqls):
            try:
                result = fetch_query_results(database_conn_info, call_plsql)
                function_results1[i] = {
                    'sql': call_plsql,
                    'result': pd.DataFrame(result)
                }
            except Exception as e:
                print(f"Warning: Could not execute call statement {i}: {call_plsql}")
                print(f"Error: {e}")
                function_results1[i] = {
                    'sql': call_plsql,
                    'result': None,
                    'error': str(e)
                }
        
        # 第二次执行
        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql2)

        function_results2 = {}
        for i, call_plsql in enumerate(call_plsqls):
            try:
                result = fetch_query_results(database_conn_info, call_plsql)
                function_results2[i] = {
                    'sql': call_plsql,
                    'result': pd.DataFrame(result)
                }
            except Exception as e:
                print(f"Warning: Could not execute call statement {i}: {call_plsql}")
                print(f"Error: {e}")
                function_results2[i] = {
                    'sql': call_plsql,
                    'result': None,
                    'error': str(e)
                }

        function_same = True
        function_diff = []

        for i in range(len(call_plsqls)):
            res1 = function_results1[i]
            res2 = function_results2[i]

            if res1.get('result') is None and res2.get('result') is None:
                continue

            if res1.get('result') is None or res2.get('result') is None:
                function_same = False
                continue

            if res1.get('result').equals(res2.get('result')):
                continue

            function_same = False
        
        print("Function same:", function_same)
        return function_same
    except Exception as e:
        print("Error in compare_plsql_function")
        print(e)
        return False



"""
PostgreSQL PL/pgSQL Semantic Equivalence Checker

This module provides tools for comparing two PL/pgSQL code blocks to determine if they are
semantically equivalent, even when they differ in:
- Whitespace and formatting
- Variable names (identifiers)
- Parameter names
- Cursor names  
- Code structure spacing

The tool uses a hybrid approach:
1. Text preprocessing to normalize syntax and formatting
2. Abstract Syntax Tree (AST) parsing using sqlparse
3. Semantic comparison of normalized AST structures

Key Features:
- Handles PostgreSQL-specific syntax ($$ delimiters, LANGUAGE plpgsql, etc.)
- Abstracts away user-defined identifiers while preserving system objects
- Normalizes PostgreSQL data types and keywords
- Supports complex PL/pgSQL constructs (cursors, exceptions, loops, etc.)

Usage:
    from postgres_db_utils import is_exact_match
    
    code1 = '''CREATE OR REPLACE PROCEDURE sp(param1 text) LANGUAGE plpgsql AS $$
               DECLARE cursor1 CURSOR FOR SELECT * FROM table1;
               BEGIN ... END; $$;'''
    
    code2 = '''CREATE OR REPLACE PROCEDURE sp(param2 text) LANGUAGE plpgsql AS $$
               DECLARE cursor2 CURSOR FOR SELECT * FROM table1;  
               BEGIN ... END; $$;'''
    
    result = is_exact_match(code1, code2)  # Returns True - semantically equivalent
"""

def preprocess_plpgsql_for_ast(sql_text: str) -> str:
    """
    Preprocess PL/pgSQL text to normalize formatting and optional syntax
    before AST parsing
    """
    # Remove extra whitespace
    sql_text = re.sub(r'\s+', ' ', sql_text.strip())
    
    # Handle PostgreSQL's $$ delimiter by temporarily replacing it
    # This helps normalize the function body
    dollar_pattern = r'\$\$([^$]*)\$\$'
    sql_text = re.sub(dollar_pattern, r'<BODY>\1</BODY>', sql_text)
    
    # Normalize LANGUAGE clause - make it consistent
    sql_text = re.sub(r'\s+LANGUAGE\s+plpgsql\s+', ' LANGUAGE plpgsql ', sql_text, flags=re.IGNORECASE)
    
    # Remove optional parameter modes (IN/OUT/INOUT) from parameter declarations
    # Match: (param_name IN datatype) -> (param_name datatype)
    sql_text = re.sub(r'\(\s*(\w+)\s+(IN|OUT|INOUT)\s+(\w+)\s*\)', 
                      r'(\1 \3)', sql_text, flags=re.IGNORECASE)
    
    # Normalize PostgreSQL data types
    # TEXT, VARCHAR, INTEGER, etc.
    pg_types = ['TEXT', 'VARCHAR', 'INTEGER', 'INT', 'BIGINT', 'SMALLINT', 
                'DECIMAL', 'NUMERIC', 'REAL', 'DOUBLE PRECISION', 'BOOLEAN', 
                'DATE', 'TIME', 'TIMESTAMP', 'TIMESTAMPTZ', 'INTERVAL', 'UUID']
    
    for pg_type in pg_types:
        # Normalize type declarations
        pattern = r'\b' + re.escape(pg_type.lower()) + r'\b'
        sql_text = re.sub(pattern, pg_type, sql_text, flags=re.IGNORECASE)
    
    # Normalize variable declarations: var_name datatype; -> var_name datatype;
    sql_text = re.sub(r'(\w+)\s+(%TYPE|%ROWTYPE)', r'\1\2', sql_text, flags=re.IGNORECASE)
    
    # Normalize exception handling
    sql_text = re.sub(r'\s+EXCEPTION\s+WHEN\s+', ' EXCEPTION WHEN ', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'\s+WHEN\s+OTHERS\s+', ' WHEN OTHERS ', sql_text, flags=re.IGNORECASE)
    
    # Normalize cursor operations
    sql_text = re.sub(r'\s+CURRENT\s+OF\s+', ' CURRENT OF ', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'\s+EXIT\s+WHEN\s+', ' EXIT WHEN ', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'\s+NOT\s+FOUND\s*', ' NOT FOUND ', sql_text, flags=re.IGNORECASE)
    
    # Normalize function calls - remove spaces in function calls
    # COUNT ( * ) -> COUNT(*)
    sql_text = re.sub(r'(\w+)\s*\(\s*\*\s*\)', r'\1(*)', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'(\w+)\s*\(\s*([^)]+)\s*\)', r'\1(\2)', sql_text)
    
    # Normalize parentheses spacing
    sql_text = re.sub(r'\s*\(\s*', '(', sql_text)
    sql_text = re.sub(r'\s*\)\s*', ')', sql_text)
    
    # Normalize punctuation spacing
    sql_text = re.sub(r'\s*;\s*', ';', sql_text)
    sql_text = re.sub(r'\s*,\s*', ',', sql_text)
    
    # Normalize operators spacing
    sql_text = re.sub(r'\s*=\s*', '=', sql_text)
    sql_text = re.sub(r'\s*>\s*', '>', sql_text)
    sql_text = re.sub(r'\s*<\s*', '<', sql_text)
    sql_text = re.sub(r'\s*:=\s*', ':=', sql_text)  # PL/pgSQL assignment
    
    # Normalize string literals - preserve quoted identifiers
    sql_text = re.sub(r'\s*"\s*([^"]+)\s*"\s*', r'"\1"', sql_text)
    
    # Ensure single space around keywords
    keywords = [
        'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'LANGUAGE', 'plpgsql', 'AS',
        'DECLARE', 'BEGIN', 'END', 'IF', 'THEN', 'ELSE', 'ELSIF', 'WHILE', 'FOR', 'LOOP',
        'SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE', 'VALUES',
        'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE', 'CURSOR', 'RECORD', 'OPEN', 'CLOSE',
        'FETCH', 'EXIT', 'WHEN', 'FOUND', 'CURRENT', 'OF', 'EXCEPTION', 'RAISE', 'RETURN',
        'COMMIT', 'ROLLBACK', 'RETURNS', 'RETURN'
    ]
    
    for keyword in keywords:
        # Add space before and after keywords
        pattern = r'\b' + re.escape(keyword) + r'\b'
        sql_text = re.sub(pattern, f' {keyword} ', sql_text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces again
    sql_text = re.sub(r'\s+', ' ', sql_text.strip())
    
    # Restore $$ delimiters but normalize them
    sql_text = re.sub(r'<BODY>([^<]*)</BODY>', r'$$\1$$', sql_text)
    
    return sql_text

class ASTNode:
    """Abstract Syntax Tree Node"""
    def __init__(self, node_type: str, value: Optional[str] = None, children: Optional[List['ASTNode']] = None):
        self.node_type = node_type
        self.value = value
        self.children = children or []
    
    def __repr__(self):
        if self.value and not self.children:
            return f"{self.node_type}({self.value})"
        elif self.children:
            children_str = ', '.join(str(child) for child in self.children)
            if self.value:
                return f"{self.node_type}({self.value})[{children_str}]"
            else:
                return f"{self.node_type}[{children_str}]"
        else:
            return self.node_type
    
    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        return (self.node_type == other.node_type and 
                self.value == other.value and 
                self.children == other.children)

class HybridPLpgSQLASTParser:
    """PL/pgSQL AST Parser that works on preprocessed, normalized text"""
    
    def __init__(self):
        self.keywords = {
            'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'LANGUAGE', 'plpgsql', 'AS', 
            'DECLARE', 'BEGIN', 'END', 'IF', 'THEN', 'ELSE', 'ELSIF', 'WHILE', 'FOR', 'LOOP',
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE', 'VALUES',
            'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE', 'CURSOR', 'RECORD', 'OPEN', 'CLOSE',
            'FETCH', 'EXIT', 'WHEN', 'FOUND', 'CURRENT', 'OF', 'EXCEPTION', 'RAISE', 'RETURN',
            'COMMIT', 'ROLLBACK', 'RETURNS', 'RETURN', 'VOLATILE', 'STABLE', 'IMMUTABLE',
            # PostgreSQL data types
            'TEXT', 'VARCHAR', 'INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'DECIMAL', 'NUMERIC',
            'REAL', 'DOUBLE', 'PRECISION', 'BOOLEAN', 'DATE', 'TIME', 'TIMESTAMP', 'TIMESTAMPTZ',
            'INTERVAL', 'UUID', 'JSON', 'JSONB', 'BYTEA', 'SERIAL', 'BIGSERIAL',
            # PostgreSQL specific keywords
            'PERFORM', 'GET', 'DIAGNOSTICS', 'ROWTYPE', 'TYPE', 'ARRAY', 'SLICE',
            'NOTICE', 'WARNING', 'STRICT', 'CONTINUE', 'CASE', 'USING', 'EXECUTE',
            'FOREACH', 'REVERSE', 'BY', 'CONCURRENTLY', 'CONFLICT', 'NOTHING', 'EXCLUDED'
        }
        # For variable name abstraction - track different types of identifiers
        self.table_names = set()
        self.column_names = set()
        self.variable_names = set()
        self.cursor_names = set()
        self.parameter_names = set()
    
    def is_system_object(self, name: str) -> bool:
        """Check if identifier refers to system objects that shouldn't be abstracted"""
        system_objects = {
            'FOUND', 'RECORD', 'SQLSTATE', 'SQLERRM', 'ROW_COUNT', 
            'CURRENT_USER', 'SESSION_USER', 'CURRENT_TIMESTAMP', 'NOW'
        }
        return name.upper() in system_objects
    
    def normalize_token_value(self, token) -> str:
        """Normalize token value for semantic equivalence with variable abstraction"""
        if not hasattr(token, 'value') or not token.value:
            return str(token)
        
        value = token.value.strip()
        
        # Normalize different token types
        if token.ttype in tokens.Literal.Number:
            return "<NUMBER>"
        elif token.ttype in tokens.Literal.String:
            return "<STRING>"
        elif (token.ttype in tokens.Name or 
              token.ttype in tokens.Name.Builtin or
              isinstance(token, sql.Identifier)):
            upper_value = value.upper()
            if upper_value in self.keywords:
                return upper_value
            elif self.is_system_object(value):
                return upper_value
            else:
                # Abstract away user-defined identifiers
                return "<IDENTIFIER>"
        elif token.ttype in tokens.Keyword:
            return value.upper()
        elif token.ttype in tokens.Punctuation:
            return value
        elif token.ttype in tokens.Operator:
            return value
        else:
            return value.upper()
    
    def parse_token_to_ast(self, token) -> Optional[ASTNode]:
        """Convert sqlparse token to AST node"""
        if token.is_whitespace:
            return None
            
        if isinstance(token, sql.Statement):
            return self.parse_statement(token)
        elif isinstance(token, sql.Parenthesis):
            return self.parse_parenthesis(token)
        elif isinstance(token, sql.Function):
            return self.parse_function(token)
        elif isinstance(token, sql.Identifier):
            return self.parse_identifier(token)
        elif isinstance(token, sql.IdentifierList):
            return self.parse_identifier_list(token)
        elif isinstance(token, sql.Comparison):
            return self.parse_comparison(token)
        elif isinstance(token, sql.Where):
            return self.parse_where(token)
        elif hasattr(token, 'tokens') and token.tokens:
            return self.parse_group(token)
        else:
            return self.parse_terminal(token)
    
    def parse_statement(self, stmt) -> ASTNode:
        """Parse SQL statement"""
        children = []
        for token in stmt.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        return ASTNode("STATEMENT", children=children)
    
    def parse_parenthesis(self, paren) -> ASTNode:
        """Parse parentheses group"""
        children = []
        for token in paren.tokens:
            if token.value not in ['(', ')']:
                child = self.parse_token_to_ast(token)
                if child:
                    children.append(child)
        return ASTNode("PARENTHESIS", children=children)
    
    def parse_function(self, func) -> ASTNode:
        """Parse function call"""
        children = []
        for token in func.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        return ASTNode("FUNCTION", children=children)
    
    def parse_identifier(self, ident) -> ASTNode:
        """Parse identifier"""
        normalized = self.normalize_token_value(ident)
        return ASTNode("IDENTIFIER", value=normalized)
    
    def parse_identifier_list(self, ident_list) -> ASTNode:
        """Parse identifier list"""
        children = []
        for token in ident_list.tokens:
            if token.value != ',':
                child = self.parse_token_to_ast(token)
                if child:
                    children.append(child)
        return ASTNode("IDENTIFIER_LIST", children=children)
    
    def parse_comparison(self, comp) -> ASTNode:
        """Parse comparison expression"""
        children = []
        for token in comp.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        return ASTNode("COMPARISON", children=children)
    
    def parse_where(self, where) -> ASTNode:
        """Parse WHERE clause"""
        children = []
        for token in where.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        return ASTNode("WHERE", children=children)
    
    def parse_group(self, group) -> ASTNode:
        """Parse generic token group"""
        children = []
        for token in group.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        
        # Determine group type based on first significant child
        if children:
            first_child = children[0]
            if (first_child.node_type == "TERMINAL" and 
                first_child.value and 
                isinstance(first_child.value, str)):
                value = first_child.value.upper()
                if value == "SELECT":
                    return ASTNode("SELECT_STATEMENT", children=children)
                elif value == "INSERT":
                    return ASTNode("INSERT_STATEMENT", children=children)
                elif value == "IF":
                    return ASTNode("IF_STATEMENT", children=children)
        
        return ASTNode("GROUP", children=children)
    
    def parse_terminal(self, token) -> ASTNode:
        """Parse terminal token"""
        normalized = self.normalize_token_value(token)
        return ASTNode("TERMINAL", value=normalized)
    
    def parse_sql(self, sql_text: str) -> List[ASTNode]:
        """Parse preprocessed SQL text into AST"""
        parsed = sqlparse.parse(sql_text)
        asts = []
        for stmt in parsed:
            ast = self.parse_token_to_ast(stmt)
            if ast:
                asts.append(ast)
        return asts

def compare_ast_nodes(node1: ASTNode, node2: ASTNode) -> bool:
    """Recursively compare two AST nodes"""
    # Compare node types
    if node1.node_type != node2.node_type:
        return False
    
    # Compare values
    if node1.value != node2.value:
        return False
    
    # Compare children count
    if len(node1.children) != len(node2.children):
        return False
    
    # Recursively compare children
    for child1, child2 in zip(node1.children, node2.children):
        if not compare_ast_nodes(child1, child2):
            return False
    
    return True

def is_exact_match_hybrid_plpgsql(plpgsql1: str, plpgsql2: str, debug: bool = False) -> bool:
    """
    Hybrid approach for PL/pgSQL: preprocess text for normalization, then use AST comparison
    
    Args:
        plpgsql1: First PL/pgSQL statement
        plpgsql2: Second PL/pgSQL statement  
        debug: Whether to show debug information
        
    Returns:
        True if semantically equivalent using AST comparison
    """
    if debug:
        print("=== Hybrid AST Comparison for PL/pgSQL (Preprocess + AST) ===")
        print(f"Original SQL1: {plpgsql1}")
        print(f"Original SQL2: {plpgsql2}")
    
    # Step 1: Preprocess both SQL texts
    preprocessed1 = preprocess_plpgsql_for_ast(plpgsql1)
    preprocessed2 = preprocess_plpgsql_for_ast(plpgsql2)
    
    if debug:
        print(f"Preprocessed SQL1: {preprocessed1}")
        print(f"Preprocessed SQL2: {preprocessed2}")
    
    # Step 2: Parse into ASTs
    parser = HybridPLpgSQLASTParser()
    ast1 = parser.parse_sql(preprocessed1)
    ast2 = parser.parse_sql(preprocessed2)
    
    if debug:
        print(f"AST1 count: {len(ast1)}")
        print(f"AST2 count: {len(ast2)}")
    
    # Step 3: Compare AST count
    if len(ast1) != len(ast2):
        if debug:
            print("Different number of AST trees")
        return False
    
    # Step 4: Compare each AST tree
    for i, (tree1, tree2) in enumerate(zip(ast1, ast2)):
        if debug:
            print(f"\n--- AST Tree {i+1} ---")
            print(f"Tree1: {tree1}")
            print(f"Tree2: {tree2}")
        
        if not compare_ast_nodes(tree1, tree2):
            if debug:
                print(f"AST trees {i+1} do not match")
            return False
        elif debug:
            print(f"AST trees {i+1} match!")
    
    return True

# Convenience functions for PL/pgSQL
def is_exact_match(plpgsql1: str, plpgsql2: str) -> bool:
    """Check semantic equivalence of PL/pgSQL using hybrid approach (no debug output)"""
    try:
        return is_exact_match_hybrid_plpgsql(plpgsql1, plpgsql2, debug=False)
    except Exception:
        return False

def debug_semantic_equivalence_ast(plpgsql1: str, plpgsql2: str) -> bool:
    """Check semantic equivalence of PL/pgSQL using hybrid approach (with debug output)"""
    return is_exact_match_hybrid_plpgsql(plpgsql1, plpgsql2, debug=True)

if __name__ == "__main__":

    print("=== Get Database Schema Graph Tests ===\n")
    print(get_database_schema_graph()["3d_coordinate_system_for_spatial_data_management"], "\n")

    print("=== restore_databases Tests ===\n")
    restore_databases(conn_info, host, port, user, password, ["3d_coordinate_system_for_spatial_data_management"])

    print("=== check_plsql_executability Tests ===\n")
    print(check_plsql_executability(f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0); END; $$;""",
                                    ["call sp(5);", "call sp(6);"],
                                    "3d_coordinate_system_for_spatial_data_management"))

    print("=== compare_plsql Tests ===\n")
    print(compare_plsql("3d_coordinate_system_for_spatial_data_management",
                        plsql1=f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0); END; $$;""",
                        plsql2=f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0,'2025-08-08'); END; $$;""",
                        call_plsqls=["call sp(5);", "call sp(6);"],
                        include_system_tables=True), "\n")
    print(compare_plsql("3d_coordinate_system_for_spatial_data_management",
                        plsql1=f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0); END; $$;""",
                        plsql2=f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0); END; $$;""",
                        call_plsqls=["call sp(5);", "call sp(6);"],
                        include_system_tables=True), "\n")
    
    print("=== restore_databases Tests ===\n")
    restore_databases(conn_info, host, port, user, password, ["3d_coordinate_system_for_spatial_data_management"])


    print("=== Get Table Info Tests ===\n")
    print(get_tables_info("3d_coordinate_system_for_spatial_data_management"), "\n")

    print("=== Get Database Schema Tests ===\n")
    print(get_database_schema("3d_coordinate_system_for_spatial_data_management"), "\n")

    print("=== Get All User Tables Tests ===\n")
    print(get_all_user_tables("3d_coordinate_system_for_spatial_data_management"), "\n")

    print("=== PL/pgSQL Semantic Equivalence Tests ===\n")
    # Test 1: Different cursor names - should return True
    print("Test 1: Different cursor names (should be True)")
    result1 = is_exact_match(
        """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql AS $$ 
           DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD; 
           BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cursor; END LOOP; 
           CLOSE ref_cursor; END; $$;""",
        
        """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql AS $$ 
           DECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD; 
           BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cur; END LOOP; 
           CLOSE ref_cur; END; $$;"""
    )
    print(f"Result: {result1}\n")
    
    # Test 2: Different formatting and spacing - should return True
    print("Test 2: Different formatting and spacing (should be True)")
    result2 = is_exact_match(
        """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql AS $$ 
           DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD; 
           BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cursor; END LOOP; 
           CLOSE ref_cursor; END; $$;""",
        
        """CREATE OR REPLACE PROCEDURE sp (para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql   AS $$  
           DECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD;   
           BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cur; END LOOP; 
           CLOSE ref_cur; END; $$;"""
    )
    print(f"Result: {result2}\n")

    # print("expect True")
    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) LANGUAGE plpgsql AS $$ DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD; BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cursor; END LOOP; CLOSE ref_cursor; END; $$;""",
                         f"""CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) LANGUAGE plpgsql AS $$ DECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD; BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cur; END LOOP; CLOSE ref_cur; END; $$;"""))
    
    # print("expect True")
    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) LANGUAGE plpgsql AS $$ DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD; BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cursor; END LOOP; CLOSE ref_cursor; END; $$;""",
                         f"""CREATE OR REPLACE PROCEDURE sp (para_state text, para_city text, para_bname text) LANGUAGE plpgsql   AS $$  \nDECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD;   BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cur; END LOOP; CLOSE ref_cur; END; $$;"""))