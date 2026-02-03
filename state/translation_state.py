from typing import TypedDict, List, Dict

class TranslationState(TypedDict):
    epoch: int
    dialect: str
    plsql_collection: List
    ir_description: List
    rewrite_description: List
    natural_languages: List
