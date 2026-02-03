import os
import re
import oracledb
import pandas as pd
from pathlib import Path
import sqlparse
from sqlparse import sql, tokens
from typing import Any, Dict, List, Optional
from config.common import oc_config
from tqdm import tqdm
import json

host = oc_config['host']
port = oc_config['port']
user = oc_config['user']
password = oc_config['password']
input_path = oc_config["input_path"]
service_name = oc_config['service_name']
db_schema_dict_path = oc_config["db_schema_dict_path"]

with open(oc_config["oracle_db_long_to_short"], 'r', encoding='utf-8') as f:
    oracle_db_long_to_short = json.load(f)
with open(oc_config["oracle_db_short_to_long"], 'r', encoding='utf-8') as f:
    oracle_db_short_to_long = json.load(f)

class OracleConnectionManager:
    """
    Oracle连接管理器，支持上下文管理和连接复用
    """
    def __init__(self):
        self.username = user
        self.password = password
        self.host = host
        self.port = port
        self.service_name = service_name
        self.importer = None
    
    def __enter__(self):
        """进入上下文时建立连接"""
        self.importer = OracleSchemaImporter()
        if not self.importer.connect():
            raise ConnectionError("无法连接到Oracle数据库")
        return self.importer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时断开连接"""
        if self.importer:
            self.importer.disconnect()
            self.importer = None

class OracleSchemaImporter:
    def __init__(self):
        """
        初始化Oracle连接参数
        """
        self.username = user
        self.password = password
        self.host = host
        self.port = port
        self.service_name = service_name
        self.connection = None
        
    def connect(self):
        """
        连接到Oracle数据库（需要有创建用户的权限）
        """
        try:
            # 构建连接字符串
            dsn = f"{self.host}:{self.port}/{self.service_name}"
            self.connection = oracledb.connect(
                user=self.username, 
                password=self.password, 
                dsn=dsn
            )
            return True
        except oracledb.Error as e:
            print(f"连接Oracle数据库失败: {e}")
            return False
    
    def disconnect(self):
        """
        断开数据库连接
        """
        if self.connection:
            self.connection.close()
    
    def extract_schema_name_from_filename(self, filename):
        """
        从文件名提取Schema名
        例: allergy_1.sqlite.sql -> ALLERGY_1
        """
        # 移除所有后缀
        schema_name = filename
        for suffix in ['.sqlite.sql', '.sql']:
            if schema_name.endswith(suffix):
                schema_name = schema_name[:-len(suffix)]
                break
        
        # 确保名称符合Oracle标识符规范
        schema_name = re.sub(r'[^a-zA-Z0-9_]', '_', schema_name)
        
        # Oracle用户名限制
        if len(schema_name) > 30:
            schema_name = schema_name[:30]
        
        return schema_name.lower()
    
    def create_schema(self, schema_name):
        """
        创建Oracle Schema（用户）
        """
        try:
            cursor = self.connection.cursor()
            
            # 生成密码
            password = f"{schema_name.lower()}_pwd"
            
            # 检查用户是否已存在
            cursor.execute("""
                SELECT COUNT(*) FROM all_users WHERE username = :username
            """, [schema_name])
            
            user_exists = cursor.fetchone()[0] > 0
            
            if user_exists:
                # 删除现有用户
                try:
                    cursor.execute(f"DROP USER {schema_name} CASCADE")
                except oracledb.Error as e:
                    if "does not exist" not in str(e).lower():
                        print(f"删除用户失败: {e}")
            
            # 创建用户
            cursor.execute(f"""
                CREATE USER {schema_name} IDENTIFIED BY {password}
                DEFAULT TABLESPACE USERS
                TEMPORARY TABLESPACE TEMP
            """)
            
            # 授权
            cursor.execute(f"GRANT CONNECT, RESOURCE TO {schema_name}")
            cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {schema_name}")
            
            # 提交
            self.connection.commit()
            cursor.close()
            
            return schema_name, password
            
        except oracledb.Error as e:
            print(f"创建Schema失败: {e}")
            return None, None
    
    def connect_to_schema(self, schema_name, password):
        """
        连接到指定Schema
        """
        try:
            dsn = f"{self.host}:{self.port}/{self.service_name}"
            schema_connection = oracledb.connect(
                user=schema_name, 
                password=password, 
                dsn=dsn
            )
            return schema_connection
        except oracledb.Error as e:
            print(f"连接Schema失败: {e}")
            return None
    
    def execute_sql_file_in_schema(self, sql_file, schema_name, password):
        """
        在指定Schema中执行SQL文件
        """
        # 连接到指定Schema
        schema_connection = self.connect_to_schema(schema_name, password)
        if not schema_connection:
            return False
        
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割SQL语句（以分号为分隔符）
            sql_statements = []
            statements = sql_content.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    sql_statements.append(statement)
            
            if not sql_statements:
                print(f"SQL文件无效: 没有找到有效的SQL语句")
                schema_connection.close()
                return False
            
            cursor = schema_connection.cursor()
            success_count = 0
            
            for i, sql in enumerate(sql_statements):
                try:
                    cursor.execute(sql)
                    success_count += 1
                except oracledb.Error as e:
                    error_msg = str(e)
                    # 忽略常见的无害错误
                    if "table or view does not exist" in error_msg.lower() and "drop" in sql.lower():
                        continue
                    elif "name is already used by an existing object" in error_msg.lower():
                        # 提取表名
                        table_match = re.search(r'CREATE TABLE\s+"?(\w+)"?', sql, re.IGNORECASE)
                        if table_match:
                            table_name = table_match.group(1)
                            try:
                                cursor.execute(f"DROP TABLE {table_name} CASCADE CONSTRAINTS")
                                cursor.execute(sql)
                                success_count += 1
                            except Exception as drop_error:
                                print(f"重新创建表失败 {table_name}: {drop_error}")
                        continue
                    else:
                        print(f"SQL执行失败 [{i+1}]: {e}")
                        continue
            
            # 提交事务
            schema_connection.commit()
            cursor.close()
            schema_connection.close()
            
            return success_count > 0
            
        except Exception as e:
            print(f"执行SQL文件失败: {e}")
            if schema_connection:
                schema_connection.close()
            return False
    
    def recreate_schema_from_sql(self, schema_name, sql_file_path):
        """
        重新创建Oracle Schema - 先删除后创建
        
        Args:
            schema_name (str): Schema名称（用户名）
            sql_file_path (str): SQL文件路径
        
        Returns:
            tuple: (success, schema_name, password) 成功标志、schema名、密码
        """
        try:
            cursor = self.connection.cursor()
            
            # 生成密码
            password = f"{schema_name.lower()}_pwd"
            
            # 1. 检查并删除现有Schema
            cursor.execute("""
                SELECT COUNT(*) FROM all_users WHERE username = :username
            """, [schema_name])
            
            user_exists = cursor.fetchone()[0] > 0
            
            if user_exists:
                try:
                    cursor.execute(f"DROP USER {schema_name} CASCADE")
                except oracledb.Error as e:
                    if "does not exist" not in str(e).lower():
                        print(f"删除Schema失败: {e}")
                        return False, None, None
            
            # 2. 创建新的Schema
            cursor.execute(f"""
                CREATE USER {schema_name} IDENTIFIED BY {password}
                DEFAULT TABLESPACE USERS
                TEMPORARY TABLESPACE TEMP
            """)
            
            # 3. 授权
            cursor.execute(f"GRANT CONNECT, RESOURCE TO {schema_name}")
            cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {schema_name}")
            
            # 提交创建用户的操作
            self.connection.commit()
            cursor.close()
            
            # 4. 执行SQL文件导入数据
            if self._execute_sql_file_in_schema(sql_file_path, schema_name, password):
                return True, schema_name, password
            else:
                print(f"Schema数据导入失败")
                return False, schema_name, password
                
        except oracledb.Error as e:
            print(f"重新创建Schema失败: {e}")
            return False, None, None

    def _execute_sql_file_in_schema(self, sql_file_path, schema_name, password):
        """
        在指定Schema中执行SQL文件（内部方法）
        
        Args:
            sql_file_path (str): SQL文件路径
            schema_name (str): Schema名称
            password (str): Schema密码
        
        Returns:
            bool: 执行成功标志
        """
        # 连接到指定Schema
        schema_connection = self.connect_to_schema(schema_name, password)
        if not schema_connection:
            return False
        
        try:
            # 读取SQL文件
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割SQL语句
            sql_statements = []
            statements = sql_content.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    sql_statements.append(statement)
            
            if not sql_statements:
                print(f"SQL文件无效: 没有找到有效的SQL语句")
                schema_connection.close()
                return False
            
            cursor = schema_connection.cursor()
            success_count = 0
            
            # 执行每条SQL语句
            for i, sql in enumerate(sql_statements):
                try:
                    cursor.execute(sql)
                    success_count += 1
                except oracledb.Error as e:
                    error_msg = str(e)
                    # 忽略常见的无害错误
                    if "table or view does not exist" in error_msg.lower() and "drop" in sql.lower():
                        continue
                    elif "name is already used by an existing object" in error_msg.lower():
                        # 提取表名并重建
                        table_match = re.search(r'CREATE TABLE\s+"?(\w+)"?', sql, re.IGNORECASE)
                        if table_match:
                            table_name = table_match.group(1)
                            try:
                                cursor.execute(f"DROP TABLE {table_name} CASCADE CONSTRAINTS")
                                cursor.execute(sql)
                                success_count += 1
                            except Exception as drop_error:
                                print(f"重新创建表失败 {table_name}: {drop_error}")
                        continue
                    else:
                        print(f"SQL执行失败 [{i+1}]: {e}")
                        continue
            
            # 提交事务
            schema_connection.commit()
            cursor.close()
            schema_connection.close()
            
            return success_count > 0
            
        except Exception as e:
            print(f"执行SQL文件失败: {e}")
            if schema_connection:
                schema_connection.close()
            return False

    def import_file_as_schema(self, sql_file):
        """
        为单个SQL文件创建Schema并导入数据
        """
        sql_path = Path(sql_file)
        
        if not sql_path.exists():
            print(f"文件不存在: {sql_file}")
            return False, None, None
        
        # 从文件名提取Schema名
        schema_name = self.extract_schema_name_from_filename(sql_path.name)
        
        # 1. 创建Schema
        schema_name, password = self.create_schema(schema_name)
        if not schema_name:
            return False, None, None
        
        # 2. 在Schema中导入数据
        success = self.execute_sql_file_in_schema(sql_path, schema_name, password)
        
        if success:
            print(f"Schema创建成功: {schema_name}")
            return True, schema_name, password
        else:
            print(f"Schema导入失败: {schema_name}")
            return False, schema_name, password
    
    def import_directory_as_schemas(self, sql_directory):
        """
        为目录中的每个SQL文件创建独立的Schema
        """
        sql_path = Path(sql_directory)
        
        if not sql_path.exists():
            print(f"目录不存在: {sql_directory}")
            return False
        
        sql_files = list(sql_path.glob("*.sql"))
        
        if not sql_files:
            print(f"目录中没有找到SQL文件: {sql_directory}")
            return False
        
        success_count = 0
        created_schemas = []
        
        for sql_file in sorted(sql_files):
            schema_name = self.extract_schema_name_from_filename(sql_file.name)
            
            # 1. 创建Schema
            created_schema, password = self.create_schema(schema_name)
            if not created_schema:
                continue
            
            # 2. 在Schema中导入数据
            if self.execute_sql_file_in_schema(sql_file, created_schema, password):
                success_count += 1
                created_schemas.append({
                    'file': sql_file.name,
                    'schema': created_schema,
                    'password': password,
                    'connection_string': f"{created_schema}/{password}@{self.host}:{self.port}/{self.service_name}"
                })
        
        print(f"批量导入完成: {success_count}/{len(sql_files)} 成功")
        
        if created_schemas:
            for schema_info in created_schemas:
                print(f"Schema: {schema_info['schema']} | 连接: {schema_info['connection_string']}")
        
        return success_count == len(sql_files)


def get_tables_info(database_name):
    """
    获取Oracle指定schema下的数据表信息
    
    Args:
        schema_name: Oracle schema名称
        host: Oracle数据库主机地址
        port: Oracle数据库端口号
        
    Returns:
        dict: 表名为key，列信息列表为value的字典
        格式: {'table_name': [('column_name', 'data_type'), ...]}
    """

    schema_name=database_name.upper()
    tables_info = {}
    
    try:
        # 建立连接 - oracledb支持直接传入参数
        with oracledb.connect(user=user, password=password, 
                             host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 查询指定schema下的所有表名
                print(conn)
                cur.execute("""
                    SELECT table_name 
                    FROM all_tables 
                    WHERE owner = :schema_name
                    ORDER BY table_name
                """, schema_name=schema_name.upper())
                
                result = cur.fetchall()
                table_names = [table_name[0] for table_name in result]
                
                # 获取每个表的列信息
                for table_name in table_names:
                    cur.execute("""
                        SELECT column_name, data_type
                        FROM all_tab_columns 
                        WHERE owner = :schema_name AND table_name = :table_name
                        ORDER BY column_id
                    """, schema_name=schema_name.upper(), table_name=table_name)
                    
                    result = cur.fetchall()
                    # 直接使用Oracle原始数据类型，不做任何转换
                    tables_info[table_name] = [(col[0], col[1]) for col in result]
                    
    except oracledb.Error as e:
        print(f"Oracle数据库连接或查询出错: {e}")
        return {}
        
    return tables_info

def get_database_schema(database_name):
    """获取数据库schema信息，返回符合DatabaseSchema类型的字典
    
    Args:
        database_name: Oracle schema名称
        
    Returns:
        Dict包含:
        - table_names: List[str] - 所有表名列表
        - tables: Dict[str, List[str]] - 表名到列名列表的映射
    """
    schema_name = database_name.upper()
    table_names = []
    tables = {}
    
    try:
        # 建立连接
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 获取所有表名
                cur.execute("""
                    SELECT table_name 
                    FROM all_tables 
                    WHERE owner = :schema_name
                    ORDER BY table_name
                """, schema_name=schema_name)
                
                result = cur.fetchall()
                table_names = [table_name[0] for table_name in result]
                
                # 获取每个表的列名
                for table_name in table_names:
                    cur.execute("""
                        SELECT column_name
                        FROM all_tab_columns 
                        WHERE owner = :schema_name AND table_name = :table_name
                        ORDER BY column_id
                    """, schema_name=schema_name, table_name=table_name)
                    
                    result = cur.fetchall()
                    tables[table_name] = [column_name[0] for column_name in result]
                    
    except oracledb.Error as e:
        print(f"Oracle数据库连接或查询出错: {e}")
        return {
            'table_names': [],
            'tables': {}
        }
    
    return {
        'table_names': table_names,
        'tables': tables
    }

def _get_foreign_key_relations(database_name):
    """获取数据库中所有表的外键关系
    
    Args:
        database_name: Oracle schema名称
        
    Returns:
        Dict[str, List[str]]: 字典，键为表名，值为与该表有外键关联的其他表名列表
    """
    schema_name = database_name.upper()
    foreign_key_relations = {}
    
    try:
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 获取所有用户表
                cur.execute("""
                    SELECT table_name
                    FROM all_tables
                    WHERE owner = :schema_name
                    AND table_name NOT LIKE 'BIN$%'
                    ORDER BY table_name
                """, {'schema_name': schema_name})
                tables = [row[0] for row in cur.fetchall()]
                
                # 初始化所有表的外键关系列表
                for table in tables:
                    foreign_key_relations[table] = []
                
                # 查询所有外键关系
                # 在Oracle中，外键约束类型为'R' (Referential)
                cur.execute("""
                    SELECT
                        a.table_name AS from_table,
                        c_pk.table_name AS to_table
                    FROM all_constraints a
                    JOIN all_constraints c_pk 
                        ON a.r_constraint_name = c_pk.constraint_name
                        AND a.r_owner = c_pk.owner
                    WHERE a.constraint_type = 'R'
                        AND a.owner = :schema_name
                        AND c_pk.owner = :schema_name
                    ORDER BY a.table_name
                """, {'schema_name': schema_name})
                
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


def get_detailed_database_schema_oracle(database_name, sample_limit=3):
    """获取Oracle数据库的详细schema信息，适合text2sql任务
    
    Args:
        database_name: Oracle schema名称
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
                                'full_data_type': str,
                                'data_length': int,
                                'data_precision': int,
                                'data_scale': int,
                                'nullable': str,
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
    schema_name = database_name.upper()
    
    try:
        # 建立连接
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 获取所有表名
                cur.execute("""
                    SELECT table_name 
                    FROM all_tables 
                    WHERE owner = :schema_name
                    AND table_name NOT LIKE 'BIN$%'
                    ORDER BY table_name
                """, schema_name=schema_name)
                
                table_names = [row[0] for row in cur.fetchall()]
                
                schema_dict = {
                    'database_name': database_name,
                    'tables': {},
                    'relationships': []
                }
                
                formatted_parts = []
                
                for table_name in table_names:
                    # 获取列详细信息
                    cur.execute("""
                        SELECT 
                            atc.column_name,
                            atc.data_type,
                            atc.data_length,
                            atc.data_precision,
                            atc.data_scale,
                            atc.nullable,
                            CASE 
                                WHEN acc.constraint_type = 'P' THEN 'PRIMARY KEY'
                                ELSE NULL 
                            END as constraint_type,
                            acc.comments as column_comment
                        FROM all_tab_columns atc
                        LEFT JOIN all_col_comments acc 
                            ON atc.owner = acc.owner 
                            AND atc.table_name = acc.table_name 
                            AND atc.column_name = acc.column_name
                        LEFT JOIN (
                            SELECT 
                                acc.column_name,
                                ac.constraint_type
                            FROM all_cons_columns acc
                            JOIN all_constraints ac 
                                ON acc.owner = ac.owner 
                                AND acc.constraint_name = ac.constraint_name
                            WHERE ac.owner = :schema_name 
                                AND acc.table_name = :table_name
                                AND ac.constraint_type IN ('P', 'U')
                        ) acc ON atc.column_name = acc.column_name
                        WHERE atc.owner = :schema_name 
                            AND atc.table_name = :table_name
                        ORDER BY atc.column_id
                    """, schema_name=schema_name, table_name=table_name)
                    
                    columns_info = cur.fetchall()
                    
                    # 数据采样
                    try:
                        cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {database_name}")
                        cur.execute(f"""
                            SELECT * FROM {table_name} 
                            WHERE ROWNUM <= {sample_limit}
                        """)
                        sample_data = cur.fetchall()
                        column_names = [desc[0] for desc in cur.description] if cur.description else []
                    except Exception as e:
                        print(f"无法采样表 {table_name} 的数据: {e}")
                        sample_data = []
                        column_names = []
                    
                    # 为每个列收集样例数据
                    column_examples = {}
                    if sample_data:
                        for i, row in enumerate(sample_data):
                            for j, col_name in enumerate(column_names):
                                if j < len(row):
                                    if col_name not in column_examples:
                                        column_examples[col_name] = []
                                    value_example = str(row[j]) if row[j] is not None else "NULL"
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
                    
                    # 构建格式化的表描述（保持原有功能）
                    table_schema = f"Table: {table_name}\n"
                    table_schema += "Columns:\n"
                    
                    for col in columns_info:
                        (column_name, data_type, data_length, data_precision, 
                         data_scale, nullable, constraint_type, col_comment) = col
                        
                        # 构建完整的数据类型
                        full_data_type = data_type
                        if data_type in ['VARCHAR2', 'CHAR', 'RAW']:
                            full_data_type += f"({data_length})"
                        elif data_type in ['NUMBER']:
                            if data_precision and data_scale:
                                full_data_type += f"({data_precision},{data_scale})"
                            elif data_precision:
                                full_data_type += f"({data_precision})"
                        
                        # 添加到字典
                        column_dict = {
                            'name': column_name,
                            'data_type': data_type,
                            'full_data_type': full_data_type,
                            'data_length': data_length,
                            'data_precision': data_precision,
                            'data_scale': data_scale,
                            'nullable': nullable,
                            'constraint_type': constraint_type,
                            'comment': col_comment,
                            'examples': column_examples.get(column_name, [])[:5]  # 最多5个样例
                        }
                        table_info['columns'].append(column_dict)
                        
                        # 构建格式化字符串
                        col_info = f"  - {column_name} ({full_data_type})"
                        
                        if constraint_type == 'PRIMARY KEY':
                            col_info += " PRIMARY KEY"
                        
                        if nullable == 'N':
                            col_info += " NOT NULL"
                        
                        if column_name in column_examples and column_examples[column_name]:
                            examples = column_examples[column_name][:5]
                            col_info += f" examples: {examples}"
                        
                        if col_comment:
                            col_info += f" - {col_comment}"
                        
                        table_schema += col_info + "\n"
                    
                    # 保存表信息到字典
                    schema_dict['tables'][table_name] = table_info
                    formatted_parts.append(table_schema)
                
                # 外键关系部分
                relationship_summary = "\nTable Relationships:\n"
                all_relationships = []
                
                for table_name in table_names:
                    cur.execute("""
                        SELECT
                            acc.column_name,
                            acc_child.table_name AS foreign_table_name,
                            acc_child.column_name AS foreign_column_name
                        FROM all_constraints ac
                        JOIN all_cons_columns acc 
                            ON ac.owner = acc.owner 
                            AND ac.constraint_name = acc.constraint_name
                        JOIN all_constraints ac_parent 
                            ON ac.r_constraint_name = ac_parent.constraint_name
                            AND ac.r_owner = ac_parent.owner
                        JOIN all_cons_columns acc_child 
                            ON ac_parent.owner = acc_child.owner 
                            AND ac_parent.constraint_name = acc_child.constraint_name
                        WHERE ac.owner = :schema_name 
                            AND ac.constraint_type = 'R'
                            AND ac.table_name = :table_name
                    """, schema_name=schema_name, table_name=table_name)
                    
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
                
                # 去重并排序
                all_relationships = sorted(list(set(all_relationships)))
                
                if all_relationships:
                    relationship_summary += "\n".join(all_relationships)
                else:
                    relationship_summary += "No foreign key relationships found."
                
                formatted_parts.append(relationship_summary)
                
                # 添加格式化字符串到字典
                schema_dict['formatted_string'] = "\n" + "="*80 + "\n" + "\n".join(formatted_parts) + "\n" + "="*80 + "\n"
                
    except oracledb.Error as e:
        error_msg = f"Oracle数据库连接或查询出错: {e}"
        print(error_msg)
        return {'error': error_msg}
    
    return schema_dict

# 辅助函数：从字典重新生成特定表的prompt（Oracle专用）
def generate_schema_prompt_from_dict(schema_dict, table_names=None):
    """从Oracle schema字典生成特定表的prompt字符串
    
    Args:
        schema_dict: get_detailed_database_schema_oracle返回的字典
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
            col_info = f"  - {col['name']} ({col['full_data_type']})"
            
            if col['constraint_type'] == 'PRIMARY KEY':
                col_info += " PRIMARY KEY"
            
            if col['nullable'] == 'N':
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


def get_database_schema_graph():
    """获取所有Oracle数据库的schema图谱
    
    Returns:
        Dict: 格式为 {
            "数据库名1": {
                "tables": ["table1", "table2"],
                "table1": [与table1有外键相连的table名]
            },
            ...
        }
    """
    # 使用与PostgreSQL相同的路径变量名
    db_schema_graph_path = oc_config.get("db_schema_graph_path", "data/oracle_db_schema_graph.json")
    
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
    """获取所有 Oracle SQL 数据库的schema JSON

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
    
    for sql_file in tqdm(sql_files[:10]):
        # 从文件名中提取数据库名（去除.sql后缀）
        database_name = sql_file[:-4]
        
        try:
            print(f"Processing database: {database_name}")
            db_schema_dict[database_name] = get_detailed_database_schema_oracle(database_name)
                
        except Exception as e:
            print(f"Error processing database {database_name}: {e}")
            continue
    
    # 保存至db_schema_dict_path
    try:
        os.makedirs(os.path.dirname(db_schema_dict_path), exist_ok=True)
        with open(db_schema_dict_path, 'w', encoding='utf-8') as f:
            json.dump(db_schema_dict, f, ensure_ascii=False, indent=2)
        print(f"Database schema dict saved to {db_schema_dict_path}")
    except Exception as e:
        print(f"Error saving database schema dict: {e}")
    
    return db_schema_dict

def get_all_user_tables(database_name):
    schema_name = database_name.upper()
    
    with oracledb.connect(user=user, password=password, 
                          host=host, port=port, service_name=service_name) as conn:
        with conn.cursor() as cur:
            # 获取指定schema中的所有用户表（排除系统表和回收站表）
            cur.execute("""
                SELECT table_name
                FROM all_tables
                WHERE owner = :schema_name
                AND table_name NOT LIKE 'BIN$%'
                ORDER BY table_name
            """, {'schema_name': schema_name})
            
            result = cur.fetchall()
            return [table_name[0] for table_name in result]

def get_important_system_tables():
    """返回需要监控的重要系统表列表"""
    return [
        'all_constraints',          # 约束信息  
        'all_triggers',             # 触发器信息
        'all_sequences',            # 序列信息
        'all_views'                 # 视图信息
    ]

def fetch_system_table_data(system_table):
    """获取系统表数据，处理可能的权限或存在性问题"""
    try:
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 检查表/视图是否存在（检查all_tables和all_views）
                cur.execute("""
                    SELECT CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM all_tables 
                            WHERE table_name = :table_name
                        ) THEN 1
                        WHEN EXISTS (
                            SELECT 1 FROM all_views 
                            WHERE view_name = :table_name
                        ) THEN 1
                        ELSE 0
                    END as table_exists
                    FROM dual
                """, {'table_name': system_table.upper()})
                
                if not cur.fetchone()[0]:
                    return None
                
                # 尝试查询系统表数据
                # Oracle需要显式指定要排序的列，这里使用ROWNUM作为默认排序
                cur.execute(f"""
                    SELECT * FROM {system_table} 
                    WHERE ROWNUM <= 100
                    ORDER BY 1
                """)
                result = cur.fetchall()
                return result
                
    except Exception as e:
        print(f"Warning: Could not fetch data from system table {system_table}: {e}")
        return None

def cleanup_plsql_objects(plsql_code, database_name):
    """清理PL/SQL代码中可能创建的函数、存储过程和触发器（Oracle版本）"""
    database_name = database_name.upper()
    try:
        recreate_database_with_context(database_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {database_name}")
                
                # 分析PL/SQL代码，识别创建的对象
                objects_to_drop = analyze_plsql_objects(plsql_code)
                
                # 清理识别到的对象
                for obj_type, obj_name in objects_to_drop:
                    if obj_type == 'function':
                        cur.execute(f"DROP FUNCTION {obj_name}")
                    elif obj_type == 'procedure':
                        cur.execute(f"DROP PROCEDURE {obj_name}")
                    elif obj_type == 'trigger':
                        cur.execute(f"DROP TRIGGER {obj_name}")
                
                conn.commit()
                
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
        ('function', r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', re.IGNORECASE),
        ('procedure', r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', re.IGNORECASE),
        ('trigger', r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', re.IGNORECASE)
    ]
    
    for obj_type, pattern, flags in patterns:
        matches = re.finditer(pattern, plsql_code, flags)
        objects.extend((obj_type, match.group(1)) for match in matches)
    
    return objects

def recreate_database_with_context(database_name):
    """
    使用上下文管理器的数据库重建函数（推荐使用）
    
    Args:
        database_name (str): 数据库名称
    
    Returns:
        tuple: (success, schema_name, password)
    """
    schema_name = database_name.upper()
    
    try:
        with OracleConnectionManager() as importer:
            success, schema_name, password = importer.recreate_schema_from_sql(
                schema_name=schema_name,
                sql_file_path=os.path.join(input_path, database_name.lower() + ".sql")
            )
            
            return success, schema_name, password
    except ConnectionError as e:
        print(f"连接错误: {e}")
        return False, None, None
    except Exception as e:
        print(f"操作失败: {e}")
        return False, None, None

def recreate_databases_with_context(database_names):
    """
    使用上下文管理器的批量数据库重建函数（推荐使用）
    
    Args:
        database_names (list): 数据库名称列表
    
    Returns:
        dict: {database_name: (success, schema_name, password)}
    """
    results = {}
    
    try:
        with OracleConnectionManager() as importer:
            for database_name in database_names:
                database_name = database_name.upper()
                
                success, schema_name, password = importer.recreate_schema_from_sql(
                    schema_name=database_name,
                    sql_file_path=os.path.join(input_path, database_name.lower() + ".sql")
                )
                
                results[database_name] = (success, schema_name, password)
        
        return results
    except ConnectionError as e:
        print(f"连接错误: {e}")
        return {name: (False, None, None) for name in database_names}
    except Exception as e:
        print(f"操作失败: {e}")
        return {name: (False, None, None) for name in database_names}

def check_plsql_executability(generated_plsql, call_plsqls, database_name):
    database_name = database_name.upper()
    execution_error = None
    try:
        recreate_database_with_context(database_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {database_name}")
                cur.call_timeout = 2 * 1000  # timeout单位为毫秒
                cur.execute(generated_plsql)
                for call in call_plsqls:
                    cur.execute(call)
                conn.commit()
    except Exception as e:
        execution_error = str(e)
    
    return execution_error

def fetch_query_results(query):
    with oracledb.connect(user=user, password=password, 
                          host=host, port=port, service_name=service_name) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            return result
        
def compare_plsql(schema_name, plsql1, plsql2, call_plsqls, include_system_tables):
    """
    比较两个PL/SQL代码的执行结果
    
    Args:
        schema_name: Oracle schema名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否包含系统表比较
    
    Returns:
        True or False
    """
    try:
        schema_name = schema_name.upper()

        # 获取所有用户表
        all_user_tables = get_all_user_tables(schema_name)
        print(f"Found {len(all_user_tables)} user tables to compare: {all_user_tables}")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Will compare {len(important_system_tables)} system tables")
        
        # 第一次执行
        recreate_database_with_context(schema_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql1)
                for call in call_plsqls:
                    cur.execute(call)
                conn.commit()

        # 收集第一次执行后的数据
        user_tables_results1 = {}
        system_tables_results1 = {}
        
        # 获取所有用户表数据
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        select_query = f'SELECT * FROM {table} ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        user_tables_results1[table] = pd.DataFrame(result)
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        user_tables_results1[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results1[sys_table] = fetch_system_table_data(sys_table)
        
        # 第二次执行
        recreate_database_with_context(schema_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql2)
                for call in call_plsqls:
                    cur.execute(call)
                conn.commit()

        # 收集第二次执行后的数据
        user_tables_results2 = {}
        system_tables_results2 = {}
        
        # 获取所有用户表数据
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        select_query = f'SELECT * FROM {table} ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        user_tables_results2[table] = pd.DataFrame(result)
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        user_tables_results2[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results2[sys_table] = fetch_system_table_data(sys_table)

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


def will_change_data(schema_name, plsql_code, call_plsqls, include_system_tables=False):
    """
    检测执行PL/SQL代码和调用语句是否会改变数据库数据
    
    Args:
        schema_name: Oracle schema名称
        plsql_code: 要检测的PL/SQL代码
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否包含系统表检测
    
    Returns:
        dict: 包含检测结果的字典
    """
    try:
        schema_name = schema_name.upper()

        # 获取所有用户表
        all_user_tables = get_all_user_tables(schema_name)
        print(f"Monitoring {len(all_user_tables)} user tables: {all_user_tables}")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Monitoring {len(important_system_tables)} system tables")
        
        # 记录执行前的数据状态
        before_user_tables_data = {}
        before_system_tables_data = {}
        
        # 获取执行前的用户表数据
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        # 获取表结构和数据
                        select_query = f'SELECT * FROM {table} ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        before_user_tables_data[table] = {
                            'data': pd.DataFrame(result),
                            'row_count': len(result)
                        }
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        before_user_tables_data[table] = None
        
        # 获取执行前的系统表数据
        for sys_table in important_system_tables:
            before_system_tables_data[sys_table] = fetch_system_table_data(sys_table)
        
        # 执行PL/SQL代码
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                try:
                    # 执行主PL/SQL代码
                    cur.execute(plsql_code)
                    
                    # 执行调用语句
                    for call in call_plsqls:
                        cur.execute(call)
                    
                    conn.commit()
                    
                except Exception as e:
                    print(f"Error during PL/SQL execution: {e}")
                    # 发生错误时回滚
                    conn.rollback()
                    return {
                        'will_change_data': False,
                        'error': str(e),
                        'execution_successful': False
                    }
        
        # 记录执行后的数据状态
        after_user_tables_data = {}
        after_system_tables_data = {}
        
        # 获取执行后的用户表数据
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        select_query = f'SELECT * FROM {table} ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        after_user_tables_data[table] = {
                            'data': pd.DataFrame(result),
                            'row_count': len(result)
                        }
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        after_user_tables_data[table] = None
        
        # 获取执行后的系统表数据
        for sys_table in important_system_tables:
            after_system_tables_data[sys_table] = fetch_system_table_data(sys_table)
        
        # 比较数据变化
        user_tables_changed = []
        user_tables_changes_detail = {}
        
        for table in all_user_tables:
            before_data = before_user_tables_data.get(table)
            after_data = after_user_tables_data.get(table)
            
            if before_data is None or after_data is None:
                # 如果表访问有问题，标记为可能变化
                user_tables_changed.append(table)
                user_tables_changes_detail[table] = "Table access error"
                continue
            
            # 比较行数变化
            if before_data['row_count'] != after_data['row_count']:
                user_tables_changed.append(table)
                user_tables_changes_detail[table] = f"Row count changed: {before_data['row_count']} -> {after_data['row_count']}"
                continue
            
            # 比较数据内容变化
            if not before_data['data'].equals(after_data['data']):
                user_tables_changed.append(table)
                user_tables_changes_detail[table] = "Data content changed"
        
        # 比较系统表变化
        system_tables_changed = []
        system_tables_changes_detail = {}
        
        if include_system_tables:
            for sys_table in important_system_tables:
                before_data = before_system_tables_data.get(sys_table)
                after_data = after_system_tables_data.get(sys_table)
                
                if before_data is None or after_data is None:
                    system_tables_changed.append(sys_table)
                    system_tables_changes_detail[sys_table] = "Table access error"
                elif before_data != after_data:
                    system_tables_changed.append(sys_table)
                    system_tables_changes_detail[sys_table] = "System table data changed"
        
        # 判断是否改变了数据
        will_change = len(user_tables_changed) > 0 or len(system_tables_changed) > 0
        
        result = {
            'will_change_data': will_change,
            'execution_successful': True,
            'user_tables_changed': user_tables_changed,
            'system_tables_changed': system_tables_changed,
            'user_tables_changes_detail': user_tables_changes_detail,
            'system_tables_changes_detail': system_tables_changes_detail,
            'total_user_tables_monitored': len(all_user_tables),
            'total_system_tables_monitored': len(important_system_tables),
            'changed_user_tables_count': len(user_tables_changed),
            'changed_system_tables_count': len(system_tables_changed)
        }
        
        print(f"Data change detection result: {result['will_change_data']}")
        if will_change:
            print(f"Changed user tables: {user_tables_changed}")
            if system_tables_changed:
                print(f"Changed system tables: {system_tables_changed}")
        
        return result
    
    except Exception as e:
        print(f"Error in will_change_data: {e}")
        return {
            'will_change_data': False,
            'error': str(e),
            'execution_successful': False
        }


def compare_plsql_function(schema_name, plsql1, plsql2, call_plsqls):
    """
    比较两个PL/SQL函数代码在Oracle中的执行结果
    
    Args:
        schema_name: Oracle schema名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
    
    Returns:
        True or False
    """
    try:
        schema_name = schema_name.upper()
        
        # 第一次执行
        recreate_database_with_context(schema_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql1)
                conn.commit()

        # 收集第一次执行的结果
        function_results1 = {}
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for i, call_plsql in enumerate(call_plsqls):
                    try:
                        cur.execute(call_plsql)
                        result = cur.fetchall()
                        
                        # 获取列名
                        col_names = [desc[0] for desc in cur.description] if cur.description else []
                        
                        if col_names:
                            function_results1[i] = {
                                'sql': call_plsql,
                                'result': pd.DataFrame(result, columns=col_names)
                            }
                        else:
                            # 如果没有列名（例如执行存储过程），创建默认DataFrame
                            function_results1[i] = {
                                'sql': call_plsql,
                                'result': pd.DataFrame(result) if result else pd.DataFrame()
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
        recreate_database_with_context(schema_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql2)
                conn.commit()

        # 收集第二次执行的结果
        function_results2 = {}
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for i, call_plsql in enumerate(call_plsqls):
                    try:
                        cur.execute(call_plsql)
                        result = cur.fetchall()
                        
                        # 获取列名
                        col_names = [desc[0] for desc in cur.description] if cur.description else []
                        
                        if col_names:
                            function_results2[i] = {
                                'sql': call_plsql,
                                'result': pd.DataFrame(result, columns=col_names)
                            }
                        else:
                            # 如果没有列名，创建默认DataFrame
                            function_results2[i] = {
                                'sql': call_plsql,
                                'result': pd.DataFrame(result) if result else pd.DataFrame()
                            }
                    except Exception as e:
                        print(f"Warning: Could not execute call statement {i}: {call_plsql}")
                        print(f"Error: {e}")
                        function_results2[i] = {
                            'sql': call_plsql,
                            'result': None,
                            'error': str(e)
                        }

        # 比较两次执行结果
        function_same = True
        function_diff = []
        
        for i in range(len(call_plsqls)):
            res1 = function_results1[i]
            res2 = function_results2[i]
            
            # 处理结果都为None的情况
            if res1.get('result') is None and res2.get('result') is None:
                continue
            
            # 处理一个结果为None的情况
            if res1.get('result') is None or res2.get('result') is None:
                function_same = False
                function_diff.append({
                    'index': i,
                    'sql': call_plsqls[i],
                    'reason': 'One result is None',
                    'result1_is_none': res1.get('result') is None,
                    'result2_is_none': res2.get('result') is None
                })
                continue
            
            # 比较DataFrame
            try:
                if not res1.get('result').equals(res2.get('result')):
                    function_same = False
                    function_diff.append({
                        'index': i,
                        'sql': call_plsqls[i],
                        'reason': 'DataFrame comparison failed',
                        'result1_shape': res1.get('result').shape,
                        'result2_shape': res2.get('result').shape
                    })
            except Exception as e:
                function_same = False
                function_diff.append({
                    'index': i,
                    'sql': call_plsqls[i],
                    'reason': f'Comparison error: {str(e)}'
                })
        
        # 输出详细比较结果
        print(f"Function comparison result: {function_same}")
        if function_diff:
            print(f"Differences found in {len(function_diff)} call statements:")
            for diff in function_diff:
                print(f"  Index {diff['index']}: {diff['sql'][:100]}...")
                print(f"    Reason: {diff['reason']}")
        
        return function_same
        
    except Exception as e:
        print(f"Error in compare_plsql_function: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def will_change_data_simple(schema_name, plsql_code, call_plsqls):
    """
    简化版本：只返回布尔值表示是否会改变数据
    """
    result = will_change_data(schema_name, plsql_code, call_plsqls, False)
    return result.get('will_change_data', False)


def preprocess_plsql_for_ast(sql_text: str) -> str:
    """
    Preprocess PL/SQL text to normalize formatting and optional syntax
    before AST parsing
    """
    # Remove extra whitespace
    sql_text = re.sub(r'\s+', ' ', sql_text.strip())
    
    # Remove optional IN/OUT/INOUT keywords from parameter declarations
    # Match: (param_name IN datatype) -> (param_name datatype)
    sql_text = re.sub(r'\(\s*(\w+)\s+(IN|OUT|INOUT)\s+(\w+)\s*\)', 
                      r'(\1 \3)', sql_text, flags=re.IGNORECASE)
    
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
    
    # Ensure single space around keywords
    keywords = [
        'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'IS', 'AS', 'BEGIN', 'END',
        'IF', 'THEN', 'ELSE', 'ELSIF', 'WHILE', 'FOR', 'LOOP', 'SELECT', 'FROM', 'WHERE',
        'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE', 'VALUES', 'COUNT', 'NUMBER', 'VARCHAR2',
        'AND', 'OR', 'NOT', 'NULL'
    ]
    
    for keyword in keywords:
        # Add space before and after keywords
        pattern = r'\b' + re.escape(keyword) + r'\b'
        sql_text = re.sub(pattern, f' {keyword} ', sql_text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces again
    sql_text = re.sub(r'\s+', ' ', sql_text.strip())
    
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

class HybridPLSQLASTParser:
    """PL/SQL AST Parser that works on preprocessed, normalized text"""
    
    def __init__(self):
        self.keywords = {
            'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'IS', 'AS', 'BEGIN', 'END',
            'IF', 'THEN', 'ELSE', 'ELSIF', 'WHILE', 'FOR', 'LOOP', 'SELECT', 'FROM', 'WHERE',
            'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE', 'VALUES', 'COUNT', 'NUMBER', 'VARCHAR2',
            'CHAR', 'DATE', 'TIMESTAMP', 'DECLARE', 'CURSOR', 'EXCEPTION', 'WHEN', 'OTHERS',
            'RAISE', 'RETURN', 'EXIT', 'CONTINUE', 'CASE', 'AND', 'OR', 'NOT', 'NULL', 
            'TRUE', 'FALSE', 'COMMIT', 'ROLLBACK'
        }
    
    def normalize_token_value(self, token) -> str:
        """Normalize token value for semantic equivalence"""
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
            else:
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

def is_exact_match_hybrid(plsql1: str, plsql2: str, debug: bool = False) -> bool:
    """
    Hybrid approach: preprocess text for normalization, then use AST comparison
    
    Args:
        plsql1: First PL/SQL statement
        plsql2: Second PL/SQL statement  
        debug: Whether to show debug information
        
    Returns:
        True if semantically equivalent using AST comparison
    """
    if debug:
        print("=== Hybrid AST Comparison (Preprocess + AST) ===")
        print(f"Original SQL1: {plsql1}")
        print(f"Original SQL2: {plsql2}")
    
    # Step 1: Preprocess both SQL texts
    preprocessed1 = preprocess_plsql_for_ast(plsql1)
    preprocessed2 = preprocess_plsql_for_ast(plsql2)
    
    if debug:
        print(f"Preprocessed SQL1: {preprocessed1}")
        print(f"Preprocessed SQL2: {preprocessed2}")
    
    # Step 2: Parse into ASTs
    parser = HybridPLSQLASTParser()
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

# Convenience functions
def is_exact_match(plsql1: str, plsql2: str) -> bool:
    """Check semantic equivalence using hybrid approach (no debug output)"""
    return is_exact_match_hybrid(plsql1, plsql2, debug=False)

def debug_semantic_equivalence_ast(plsql1: str, plsql2: str) -> bool:
    """Check semantic equivalence using hybrid approach (with debug output)"""
    return is_exact_match_hybrid(plsql1, plsql2, debug=True)

if __name__ == "__main__":

    short_name = oracle_db_long_to_short["netcdf_file_metadata_management"]
    print(short_name)
    long_name = oracle_db_short_to_long[short_name]
    print(long_name)

    print("=== Get Database Schema Graph Tests ===\n")
    print(get_database_schema_graph()["natural_lpm_management"], "\n")

    print("=== Get Table Info Tests ===\n")
    print(get_tables_info(short_name), "\n")
    
    print("=== Get Database Schema Tests ===\n")
    print(get_database_schema(short_name), "\n")

    print("=== Get All User Tables Tests ===\n")
    print(get_all_user_tables(short_name), "\n")

    print("=== Get Important System Tables Tests ===\n")
    print(get_important_system_tables(), "\n")

    print("=== Fetch System Table Data Tests ===\n")
    print(fetch_system_table_data("all_constraints"), "\n")

    print("=== Recreate Database with Context Tests ===\n")
    print(recreate_database_with_context(short_name), "\n")

    print("=== Recreate Databases with Context Tests ===\n")
    print(recreate_databases_with_context([short_name]), "\n")

    print("=== Check PL/SQL Executability Tests ===\n")
    print(check_plsql_executability("""CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = para_name WHERE \"attribute_id\" = para_attribute_id; END;""",
                              ["BEGIN sp(0, '666'); COMMIT; END;",
                               "BEGIN sp(1, '790'); COMMIT; END;",
                               "BEGIN sp(0, '785'); COMMIT; END;"],
                                short_name))

    print("=== Compare PL/SQL Tests ===\n")
    print(compare_plsql(short_name,
                        """CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = para_name WHERE \"attribute_id\" = para_attribute_id; END;""",
                        """CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = '111122' WHERE \"attribute_id\" = para_attribute_id; END;""",
                        ["BEGIN sp(0, '666'); COMMIT; END;",
                         "BEGIN sp(1, '790'); COMMIT; END;",
                         "BEGIN sp(0, '785'); COMMIT; END;"],
                         True))

    print(compare_plsql(short_name,
                        """CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = para_name WHERE \"attribute_id\" = para_attribute_id; END;""",
                        """CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = para_name WHERE \"attribute_id\" = para_attribute_id; END;""",
                        ["BEGIN sp(0, '666'); COMMIT; END;",
                         "BEGIN sp(1, '790'); COMMIT; END;",
                         "BEGIN sp(0, '785'); COMMIT; END;"],
                         True))

    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_Gender VARCHAR2) IS record_count NUMBER; BEGIN SELECT COUNT(*) INTO record_count FROM "coach" WHERE "Gender" = para_Gender; IF record_count = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END""",
                         f"""CREATE OR REPLACE PROCEDURE sp(para_Gender VARCHAR2) IS record_coun NUMBER; BEGIN SELECT COUNT(*) INTO record_coun FROM "coach" WHERE "Gender" = para_Gender; IF record_coun = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END"""))

    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_Gender VARCHAR2) IS record_count NUMBER; BEGIN SELECT COUNT(*) INTO record_count FROM "coach" WHERE "Gender" = para_Gender; IF record_count = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END""",
                         f"""CREATE OR REPLACE PROCEDURE sp(para_Gender IN VARCHAR2) IS record_coun NUMBER; BEGIN SELECT COUNT(*) INTO record_coun FROM "coach" WHERE "Gender" = para_Gender; IF record_coun = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END"""))