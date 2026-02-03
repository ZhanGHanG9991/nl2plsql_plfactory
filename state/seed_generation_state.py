from typing import TypedDict, List, Dict

class DatabaseSchema(TypedDict):
    table_names: List[str]
    tables: Dict[str, List[str]]

class SeedGenerationState(TypedDict, total=False):
    epoch: int
    target_plsql_number: int
    current_plsql_number: int
    dialect: str
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
    generated_plsqls: List[str]

    database_statistics: dict

    seeds: List[dict]
    stage: str
