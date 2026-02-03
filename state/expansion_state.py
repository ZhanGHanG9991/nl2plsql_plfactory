from typing import TypedDict, List, Dict, Literal

class DatabaseSchema(TypedDict):
    table_names: List[str]
    tables: Dict[str, List[str]]

class ExpansionState(TypedDict, total=False):
    epoch: int
    dialect: str
    selected_seed: List[dict]
    target_plsql_number: int
    current_plsql_number: int
    expansion_mode: str
    critical_epoch: int
    need_correction: List[bool]
    execution_info: List[str]
    critical_again: bool
    correction_experience: List[List[str]]

    selected_database_name: str
    selected_database_schema: DatabaseSchema
    selected_detailed_database_schema: dict
    selected_table_number: int
    selected_tables: List[str]
    
    expansion_plsqls: List[dict]

    database_statistics: dict

    seeds: List[dict]
    ir_plsqls: List[dict]
