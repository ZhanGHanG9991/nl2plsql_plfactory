import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.common import strong_llm_config, pg_config, oc_config
from util.llm_util import init_llm_with_random_provider, call_llm_with_retry
import util.postgres_util as postgres_util
import util.oracle_util as oracle_util
from langchain_core.messages import BaseMessage


_ORIGINAL_SECTION_PATTERN = re.compile(
    r"<original_plsql>\s*(.*?)\s*</original_plsql>", re.DOTALL | re.IGNORECASE
)
_CORRECTION_SECTION_PATTERN = re.compile(
    r"<correction_experience>\s*(.*?)\s*</correction_experience>", re.DOTALL | re.IGNORECASE
)
_CORRECTED_SECTION_PATTERN = re.compile(
    r"<corrected_plsql>\s*(.*?)\s*</corrected_plsql>", re.DOTALL | re.IGNORECASE
)

_PG_ERROR_DOC_CACHE: Optional[dict] = None
_OC_ERROR_DOC_CACHE: Optional[dict] = None


def _load_error_doc(dialect: str) -> Dict[str, Dict[str, str]]:
    global _PG_ERROR_DOC_CACHE, _OC_ERROR_DOC_CACHE
    try:
        if dialect == "postgresql":
            if _PG_ERROR_DOC_CACHE is None:
                doc_path = pg_config.get("errors_doc_path")
                if not doc_path:
                    _PG_ERROR_DOC_CACHE = {}
                else:
                    with open(doc_path, "r", encoding="utf-8") as f:
                        _PG_ERROR_DOC_CACHE = json.load(f)
            return _PG_ERROR_DOC_CACHE or {}
        if dialect == "oracle":
            if _OC_ERROR_DOC_CACHE is None:
                doc_path = oc_config.get("errors_doc_path")
                if not doc_path:
                    _OC_ERROR_DOC_CACHE = {}
                else:
                    with open(doc_path, "r", encoding="utf-8") as f:
                        _OC_ERROR_DOC_CACHE = json.load(f)
            return _OC_ERROR_DOC_CACHE or {}
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return {}


def _extract_error_code_postgres(execution_info: str) -> Optional[str]:
    if not execution_info:
        return None
    colon_idx = execution_info.find(":")
    if colon_idx >= 5:
        candidate = execution_info[colon_idx - 5:colon_idx].strip()
        if len(candidate) == 5 and re.fullmatch(r"[A-Z0-9]{5}", candidate):
            return candidate
    match = re.search(r"([A-Z0-9]{5}):", execution_info)
    if match:
        return match.group(1)
    return None


def _extract_error_code_oracle(execution_info: str) -> Optional[str]:
    if not execution_info:
        return None
    match = re.search(r"(ORA-\d{5})", execution_info)
    if match:
        return match.group(1)
    return None


def _build_error_knowledge(dialect: str, execution_info: str) -> Optional[str]:
    if not execution_info:
        return None

    if dialect == "postgresql":
        error_code = _extract_error_code_postgres(execution_info)
    elif dialect == "oracle":
        error_code = _extract_error_code_oracle(execution_info)
    else:
        return None

    if not error_code:
        return None

    error_doc = _load_error_doc(dialect)
    error_info = error_doc.get(error_code)
    if not isinstance(error_info, dict):
        return None

    subtitle = error_info.get("subtitle")
    cause = error_info.get("Cause") or error_info.get("cause")
    action = error_info.get("Action") or error_info.get("action")

    if not (subtitle and cause and action):
        return None

    return f"Error: {subtitle} \nCause: {cause} \nSolution: {action}"


def _call_review_llm_with_retry(messages, max_retries: int = 3, timeout: float = 120.0):
    """
    使用超时和重试机制调用review LLM
    
    Args:
        messages: 要发送给LLM的消息列表
        max_retries: 最大重试次数，默认3次
        timeout: 每次调用的超时时间（秒），默认120秒
    
    Returns:
        LLM响应对象
    """
    review_model_cfg = strong_llm_config.get("review_model", {})
    
    def llm_call(llm):
        """LLM调用函数"""
        return llm.invoke(messages)
    
    # 使用超时重试机制调用
    response = call_llm_with_retry(
        llm_func_factory=llm_call,
        model_config=review_model_cfg,
        max_retries=max_retries,
        timeout=timeout,
        model_name="review_model",
        verbose=True
    )
    
    return response

def _print_messages(title: str, messages: List[BaseMessage]) -> None:
    print("\n" + "=" * 80)
    print(f"【{title}】")
    print("=" * 80)
    for i, m in enumerate(messages, start=1):
        role = getattr(m, "type", None) or m.__class__.__name__.replace("Message", "").lower()
        name = getattr(m, "name", None)
        header = f"[{i}] role={role}" + (f", name={name}" if name else "")
        print(header)
        print("-" * 80)
        # LangChain 的 content 可能是 str 或 list[dict]（如带工具消息时）
        if isinstance(m.content, str):
            print(m.content)
        else:
            print(repr(m.content))
        print()
    print("=" * 80 + "\n")

_POSTGRES_FEW_SHOT_MESSAGES = [
    SystemMessage(
        content=(
            "You are an elite PostgreSQL PL/pgSQL engineer. Carefully read the provided database schema "
            "and execution feedback. Identify structural, syntactic, or data-related flaws in the "
            "PL/pgSQL snippet. Fix them by thinking step-by-step, ensuring the corrected code is "
            "executable, logically consistent, and aligned with the schema constraints. Always "
            "return results in the mandated structured format."
        )
    ),
    HumanMessage(
        content=(
            "Database Schema:\n"
            "- \"airport\"(\"Airport_ID\" INTEGER PRIMARY KEY, \"Total_Passengers\" REAL, \"International_Passengers\" REAL, \"Domestic_Passengers\" REAL)\n"
            "- \"flight\"(\"Flight_ID\" INTEGER PRIMARY KEY, \"Airport_ID\" INTEGER, \"Airline\" TEXT)\n\n"
            "Execution Feedback:\n"
            "ERROR: relation \"airports\" does not exist\n"
            "CONTEXT: SQL statement \"UPDATE airports SET \\\"TotalPassenger\\\" = ...\"\n\n"
            "Original PL/pgSQL:\n"
            "<start-sql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_Total_Passengers real, para_International_Passengers real, para_Airport_ID integer) LANGUAGE plpgsql AS $$\n"
            "BEGIN\n"
            "  UPDATE airports\n"
            "  SET \"TotalPassenger\" = para_Total_Passengers * para_Total_Passengers,\n"
            "      \"International_Passengers\" = para_International_Passengers;\n"
            "  DELETE FROM \"airport\"\n"
            "  WHERE \"Airport_ID\" = para_Airport_ID + '1';\n"
            "END;\n"
            "$$;\n"
            "<end-sql>\n\n"
            "Instructions:\n"
            "1. Diagnose the failure using the schema and execution feedback.\n"
            "2. Adjust joins, predicates, data transformations, control flow, or exception handling as needed.\n"
            "3. Return the fixed PL/pgSQL only inside <corrected_plsql> tags.\n"
            "4. Summarize the correction insights inside <correction_experience> tags.\n"
            "5. Echo the problematic PL/pgSQL in <original_plsql> tags.\n"
            "6. Do not add commentary outside the specified tags."
        )
    ),
    AIMessage(
        content=(
            "<original_plsql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_Total_Passengers real, para_International_Passengers real, para_Airport_ID integer) LANGUAGE plpgsql AS $$\n"
            "BEGIN\n"
            "  UPDATE airports\n"
            "  SET \"TotalPassenger\" = para_Total_Passengers * para_Total_Passengers,\n"
            "      \"International_Passengers\" = para_International_Passengers;\n"
            "  DELETE FROM \"airport\"\n"
            "  WHERE \"Airport_ID\" = para_Airport_ID + '1';\n"
            "END;\n"
            "$$;\n"
            "</original_plsql>\n"
            "<correction_experience>\n"
            "Switched the UPDATE target to \"airport\", restored the exact column name \"Total_Passengers\", "
            "and used numeric addition when comparing \"Airport_ID\" with the incoming parameter.\n"
            "</correction_experience>\n"
            "<corrected_plsql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_Total_Passengers real, para_International_Passengers real, para_Airport_ID integer)\n"
            "LANGUAGE plpgsql\n"
            "AS $$\n"
            "BEGIN\n"
            "  UPDATE \"airport\"\n"
            "  SET \"Total_Passengers\" = para_Total_Passengers * para_Total_Passengers - 1,\n"
            "      \"International_Passengers\" = para_International_Passengers;\n"
            "\n"
            "  DELETE FROM \"airport\"\n"
            "  WHERE \"Airport_ID\" = para_Airport_ID + 1;\n"
            "END;\n"
            "$$;\n"
            "</corrected_plsql>"
        )
    ),
    HumanMessage(
        content=(
            "Database Schema:\n"
            "- \"station\"(\"id\" INTEGER PRIMARY KEY, \"name\" TEXT, \"region\" TEXT)\n"
            "- \"weather\"(\"id\" INTEGER PRIMARY KEY, \"mean_dew_point_f\" INTEGER, \"min_dew_point_f\" INTEGER, \"mean_visibility_miles\" INTEGER)\n\n"
            "Execution Feedback:\n"
            "ERROR: relation \"weather_data\" does not exist\n"
            "CONTEXT: SQL statement \"UPDATE weather_data SET ...\"\n\n"
            "Original PL/pgSQL:\n"
            "<start-sql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_min_dew_point_f integer, para_mean_dew_point_f integer, para_mean_visibility_miles integer, para_id integer, para_name text) LANGUAGE plpgsql AS $$\n"
            "DECLARE\n"
            "  ref_cursor CURSOR FOR SELECT * FROM \"station\" WHERE \"id\" >= para_id;\n"
            "  rec RECORD;\n"
            "BEGIN\n"
            "  UPDATE weather_data\n"
            "  SET \"min_dew_point_f\" = para_min_dew_point_f\n"
            "  WHERE \"mean_dew_point_f\" = para_mean_dew_point_f;\n"
            "\n"
            "  OPEN ref_cursor;\n"
            "  LOOP\n"
            "    FETCH ref_cursor INTO rec;\n"
            "    EXIT WHEN NOT FOUND;\n"
            "    IF rec.\"name\" IS NULL THEN\n"
            "      UPDATE \"station\" SET \"name\" = para_name;\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "  CLOSE ref_cursor;\n"
            "\n"
            "  DELETE FROM \"weather\"\n"
            "  WHERE \"mean_visibility_miles\" <= para_mean_visibility_miles;\n"
            "END;\n"
            "$$;\n"
            "<end-sql>\n\n"
            "Instructions:\n"
            "1. Diagnose the failure using the schema and execution feedback.\n"
            "2. Adjust joins, predicates, data transformations, control flow, or exception handling as needed.\n"
            "3. Return the fixed PL/pgSQL only inside <corrected_plsql> tags.\n"
            "4. Summarize the correction insights inside <correction_experience> tags.\n"
            "5. Echo the problematic PL/pgSQL in <original_plsql> tags.\n"
            "6. Do not add commentary outside the specified tags."
        )
    ),
    AIMessage(
        content=(
            "<original_plsql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_min_dew_point_f integer, para_mean_dew_point_f integer, para_mean_visibility_miles integer, para_id integer, para_name text) LANGUAGE plpgsql AS $$\n"
            "DECLARE\n"
            "  ref_cursor CURSOR FOR SELECT * FROM \"station\" WHERE \"id\" >= para_id;\n"
            "  rec RECORD;\n"
            "BEGIN\n"
            "  UPDATE weather_data\n"
            "  SET \"min_dew_point_f\" = para_min_dew_point_f\n"
            "  WHERE \"mean_dew_point_f\" = para_mean_dew_point_f;\n"
            "\n"
            "  OPEN ref_cursor;\n"
            "  LOOP\n"
            "    FETCH ref_cursor INTO rec;\n"
            "    EXIT WHEN NOT FOUND;\n"
            "    IF rec.\"name\" IS NULL THEN\n"
            "      UPDATE \"station\" SET \"name\" = para_name;\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "  CLOSE ref_cursor;\n"
            "\n"
            "  DELETE FROM \"weather\"\n"
            "  WHERE \"mean_visibility_miles\" <= para_mean_visibility_miles;\n"
            "END;\n"
            "$$;\n"
            "</original_plsql>\n"
            "<correction_experience>\n"
            "Targeted the existing table \"weather\" in the UPDATE, preserved quoted identifiers for case-sensitive columns, "
            "and used WHERE CURRENT OF to limit in-place station updates while tightening the deletion predicate semantics.\n"
            "</correction_experience>\n"
            "<corrected_plsql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_min_dew_point_f integer, para_mean_dew_point_f integer, para_mean_visibility_miles integer, para_id integer, para_name text)\n"
            "LANGUAGE plpgsql\n"
            "AS $$\n"
            "DECLARE\n"
            "  ref_cursor CURSOR FOR SELECT * FROM \"station\" WHERE \"id\" > para_id;\n"
            "  rec RECORD;\n"
            "BEGIN\n"
            "  UPDATE \"weather\"\n"
            "  SET \"min_dew_point_f\" = para_min_dew_point_f\n"
            "  WHERE \"mean_dew_point_f\" = para_mean_dew_point_f;\n"
            "\n"
            "  OPEN ref_cursor;\n"
            "  LOOP\n"
            "    FETCH ref_cursor INTO rec;\n"
            "    EXIT WHEN NOT FOUND;\n"
            "    IF rec.\"name\" IS NULL THEN\n"
            "      UPDATE \"station\" SET \"name\" = para_name WHERE CURRENT OF ref_cursor;\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "  CLOSE ref_cursor;\n"
            "\n"
            "  DELETE FROM \"weather\"\n"
            "  WHERE \"mean_visibility_miles\" < para_mean_visibility_miles;\n"
            "END;\n"
            "$$;\n"
            "</corrected_plsql>"
        )
    ),
]

_ORACLE_FEW_SHOT_MESSAGES = [
    SystemMessage(
        content=(
            "You are an elite Oracle PL/SQL engineer. Carefully read the provided database schema "
            "and execution feedback. Identify structural, syntactic, or data-related flaws in the "
            "PL/SQL snippet. Fix them by thinking step-by-step, ensuring the corrected PL/SQL is "
            "executable, logically consistent, and aligned with the schema constraints. Always "
            "return results in the mandated structured format."
        )
    ),
    HumanMessage(
        content=(
            "Database Schema:\n"
            "- \"Rating\"(\"mID\" NUMBER PRIMARY KEY, \"reviewer\" VARCHAR2(100), \"stars\" NUMBER)\n\n"
            "Execution Feedback:\n"
            "ORA-00942: table or view does not exist\n"
            "Statement processed: UPDATE \"Ratings\" SET \"stars\" = para_stars / para_mID ...\n\n"
            "Original PL/SQL:\n"
            "<start-sql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_stars NUMBER, para_mID NUMBER) IS\n"
            "BEGIN\n"
            "  IF para_mID <> 0 THEN\n"
            "    UPDATE \"Ratings\"\n"
            "       SET \"stars\" = para_stars / para_mID\n"
            "     WHERE \"movie_id\" IS NULL;\n"
            "  END IF;\n"
            "\n"
            "  UPDATE \"Rating\"\n"
            "     SET stars = para_stars * para_mID\n"
            "   WHERE \"mID\" IS NOT NULL;\n"
            "END;\n"
            "<end-sql>\n\n"
            "Instructions:\n"
            "1. Diagnose the failure using the schema and execution feedback.\n"
            "2. Adjust joins, predicates, data transformations, control flow, or exception handling as needed.\n"
            "3. Return the fixed PL/SQL only inside <corrected_plsql> tags.\n"
            "4. Summarize the correction insights inside <correction_experience> tags.\n"
            "5. Echo the problematic PL/SQL in <original_plsql> tags.\n"
            "6. Do not add commentary outside the specified tags."
        )
    ),
    AIMessage(
        content=(
            "<original_plsql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_stars NUMBER, para_mID NUMBER) IS\n"
            "BEGIN\n"
            "  IF para_mID <> 0 THEN\n"
            "    UPDATE \"Ratings\"\n"
            "       SET \"stars\" = para_stars / para_mID\n"
            "     WHERE \"movie_id\" IS NULL;\n"
            "  END IF;\n"
            "\n"
            "  UPDATE \"Rating\"\n"
            "     SET stars = para_stars * para_mID\n"
            "   WHERE \"mID\" IS NOT NULL;\n"
            "END;\n"
            "</original_plsql>\n"
            "<correction_experience>\n"
            "Targeted the existing table \"Rating\", referenced oracle-quoted identifiers exactly, "
            "and preserved the conditional branch to avoid division by zero while updating NULL and non-NULL partitions separately.\n"
            "</correction_experience>\n"
            "<corrected_plsql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_stars NUMBER, para_mID NUMBER) IS\n"
            "BEGIN\n"
            "  IF para_mID <> 0 THEN\n"
            "    UPDATE \"Rating\"\n"
            "       SET \"stars\" = para_stars / para_mID\n"
            "     WHERE \"mID\" IS NULL;\n"
            "  END IF;\n"
            "\n"
            "  UPDATE \"Rating\"\n"
            "     SET \"stars\" = para_stars * para_mID\n"
            "   WHERE \"mID\" IS NOT NULL;\n"
            "END;\n"
            "</corrected_plsql>"
        )
    ),
    HumanMessage(
        content=(
            "Database Schema:\n"
            "- \"trip\"(\"id\" NUMBER PRIMARY KEY, \"duration\" NUMBER, \"end_station_name\" VARCHAR2(100))\n"
            "- \"status\"(\"station_id\" NUMBER PRIMARY KEY, \"docks_available\" NUMBER)\n"
            "- \"weather\"(\"id\" NUMBER PRIMARY KEY, \"events\" VARCHAR2(100), \"mean_wind_speed_mph\" NUMBER)\n\n"
            "Execution Feedback:\n"
            "ORA-00942: table or view does not exist\n"
            "Statement processed: UPDATE trip SET \"end_station_name\" = para_end_station_name ...\n\n"
            "Original PL/SQL:\n"
            "<start-sql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_events VARCHAR2, para_mean_wind_speed_mph NUMBER, para_duration NUMBER, para_end_station_name VARCHAR2, para_station_id NUMBER, para_docks_available NUMBER) IS\n"
            "  CURSOR ref_cursor IS SELECT * FROM \"trip\";\n"
            "  CURSOR ref_2Cursor IS SELECT * FROM status FOR UPDATE NOWAIT;\n"
            "  rec \"trip\"%ROWTYPE;\n"
            "  rec2 \"status\"%ROWTYPE;\n"
            "BEGIN\n"
            "  UPDATE weather\n"
            "     SET \"events\" = para_events\n"
            "   WHERE mean_wind_speed_mph > para_mean_wind_speed_mph;\n"
            "\n"
            "  OPEN ref_cursor;\n"
            "  LOOP\n"
            "    FETCH ref_cursor INTO rec;\n"
            "    EXIT WHEN ref_cursor%NOTFOUND;\n"
            "    IF (rec.\"duration\" > para_duration) THEN\n"
            "      UPDATE trip\n"
            "         SET \"end_station_name\" = para_end_station_name\n"
            "       WHERE \"id\" = rec.\"id\";\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "  CLOSE ref_cursor;\n"
            "\n"
            "  OPEN ref_2Cursor;\n"
            "  LOOP\n"
            "    FETCH ref_2Cursor INTO rec2;\n"
            "    EXIT WHEN ref_2Cursor%NOTFOUND;\n"
            "    IF (rec2.\"station_id\" > para_station_id) THEN\n"
            "      UPDATE status\n"
            "         SET \"docks_available\" = para_docks_available;\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "  CLOSE ref_2Cursor;\n"
            "END;\n"
            "<end-sql>\n\n"
            "Instructions:\n"
            "1. Diagnose the failure using the schema and execution feedback.\n"
            "2. Adjust joins, predicates, data transformations, control flow, or exception handling as needed.\n"
            "3. Return the fixed PL/SQL only inside <corrected_plsql> tags.\n"
            "4. Summarize the correction insights inside <correction_experience> tags.\n"
            "5. Echo the problematic PL/SQL in <original_plsql> tags.\n"
            "6. Do not add commentary outside the specified tags."
        )
    ),
    AIMessage(
        content=(
            "<original_plsql>\n"
            "CREATE OR REPLACE PROCEDURE sp(para_events VARCHAR2, para_mean_wind_speed_mph NUMBER, para_duration NUMBER, para_end_station_name VARCHAR2, para_station_id NUMBER, para_docks_available NUMBER) IS\n"
            "  CURSOR ref_cursor IS SELECT * FROM \"trip\";\n"
            "  CURSOR ref_2Cursor IS SELECT * FROM status FOR UPDATE NOWAIT;\n"
            "  rec \"trip\"%ROWTYPE;\n"
            "  rec2 \"status\"%ROWTYPE;\n"
            "BEGIN\n"
            "  UPDATE weather\n"
            "     SET \"events\" = para_events\n"
            "   WHERE mean_wind_speed_mph > para_mean_wind_speed_mph;\n"
            "\n"
            "  OPEN ref_cursor;\n"
            "  LOOP\n"
            "    FETCH ref_cursor INTO rec;\n"
            "    EXIT WHEN ref_cursor%NOTFOUND;\n"
            "    IF (rec.\"duration\" > para_duration) THEN\n"
            "      UPDATE trip\n"
            "         SET \"end_station_name\" = para_end_station_name\n"
            "       WHERE \"id\" = rec.\"id\";\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "  CLOSE ref_cursor;\n"
            "\n"
            "  OPEN ref_2Cursor;\n"
            "  LOOP\n"
            "    FETCH ref_2Cursor INTO rec2;\n"
            "    EXIT WHEN ref_2Cursor%NOTFOUND;\n"
            "    IF (rec2.\"station_id\" > para_station_id) THEN\n"
            "      UPDATE status\n"
            "         SET \"docks_available\" = para_docks_available;\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "  CLOSE ref_2Cursor;\n"
            "END;\n"
            "</original_plsql>\n"
            "<correction_experience>\n"
            "Referenced quoted identifiers for case-sensitive objects, used FOR UPDATE cursors so WHERE CURRENT OF targets the fetched rows, "
            "and kept updates scoped to the matched records in both loops while filtering weather rows with quoted column names.\n"
            "</correction_experience>\n"
            "<corrected_plsql>\n"
            "CREATE OR REPLACE PROCEDURE sp(\n"
            "  para_events             VARCHAR2,\n"
            "  para_mean_wind_speed_mph NUMBER,\n"
            "  para_duration           NUMBER,\n"
            "  para_end_station_name   VARCHAR2,\n"
            "  para_station_id         NUMBER,\n"
            "  para_docks_available    NUMBER\n"
            ") IS\n"
            "  CURSOR ref_cursor IS\n"
            "    SELECT * FROM \"trip\" FOR UPDATE;\n"
            "  CURSOR ref_2Cursor IS\n"
            "    SELECT * FROM \"status\" FOR UPDATE;\n"
            "  rec  \"trip\"%ROWTYPE;\n"
            "  rec2 \"status\"%ROWTYPE;\n"
            "BEGIN\n"
            "  UPDATE \"weather\"\n"
            "     SET \"events\" = para_events\n"
            "   WHERE \"mean_wind_speed_mph\" > para_mean_wind_speed_mph;\n"
            "\n"
            "  OPEN ref_cursor;\n"
            "  LOOP\n"
            "    FETCH ref_cursor INTO rec;\n"
            "    EXIT WHEN ref_cursor%NOTFOUND;\n"
            "    IF rec.\"duration\" > 2 THEN\n"
            "      UPDATE \"trip\"\n"
            "         SET \"end_station_name\" = para_end_station_name\n"
            "       WHERE CURRENT OF ref_cursor;\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "  CLOSE ref_cursor;\n"
            "\n"
            "  OPEN ref_2Cursor;\n"
            "  LOOP\n"
            "    FETCH ref_2Cursor INTO rec2;\n"
            "    EXIT WHEN ref_2Cursor%NOTFOUND;\n"
            "    IF rec2.\"station_id\" > para_station_id THEN\n"
            "      UPDATE \"status\"\n"
            "         SET \"docks_available\" = para_docks_available\n"
            "       WHERE CURRENT OF ref_2Cursor;\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "  CLOSE ref_2Cursor;\n"
            "END;\n"
            "</corrected_plsql>"
        )
    ),
]


def _get_base_messages(dialect: str) -> List:
    if dialect == "postgresql":
        return list(_POSTGRES_FEW_SHOT_MESSAGES)
    if dialect == "oracle":
        return list(_ORACLE_FEW_SHOT_MESSAGES)
    raise ValueError(f"Unsupported dialect '{dialect}'. Expected 'postgresql' or 'oracle'.")


def _parse_correction_record(record: str) -> Tuple[str, str, str]:
    if not record:
        raise ValueError("Empty correction record encountered.")

    original_match = _ORIGINAL_SECTION_PATTERN.search(record)
    correction_match = _CORRECTION_SECTION_PATTERN.search(record)
    corrected_match = _CORRECTED_SECTION_PATTERN.search(record)

    if not (original_match and correction_match and corrected_match):
        raise ValueError("Failed to extract sections from correction record.")

    return (
        original_match.group(1).strip(),
        correction_match.group(1).strip(),
        corrected_match.group(1).strip(),
    )


def _build_experience_messages(
    prior_experiences: Sequence[str],
    execution_info: str,
) -> List:
    messages: List = []
    for idx, record in enumerate(prior_experiences, start=1):
        try:
            original_sql, experience, corrected_sql = _parse_correction_record(record)
        except ValueError:
            continue

        messages.append(
            HumanMessage(
                content=(
                    f"Attempt {idx} - original PL/SQL that still failed:\n"
                    f"<original_plsql>\n{original_sql}\n</original_plsql>"
                )
            )
        )
        messages.append(
            AIMessage(
                content=(
                    f"<correction_experience>\n{experience}\n</correction_experience>\n"
                    f"<corrected_plsql>\n{corrected_sql}\n</corrected_plsql>\n"
                    f"Outcome: this correction attempt still failed. New execution feedback to resolve: {execution_info}"
                )
            )
        )
    return messages


def correction_agent(
    current_plsql: str,
    selected_tables: List[str],
    selected_database_schema: Optional[dict],
    execution_info: str,
    critical_epoch: int = 0,
    prior_experiences: Optional[Sequence[str]] = None,
    dialect: str = "oracle",
) -> Tuple[str, str, str]:
    messages = _get_base_messages(dialect)

    if dialect == "postgresql":
        database_schema_str = postgres_util.generate_schema_prompt_from_dict(selected_database_schema, selected_tables)
    elif dialect == "oracle":
        database_schema_str = oracle_util.generate_schema_prompt_from_dict(selected_database_schema, selected_tables)
    else:
        raise ValueError(f"Database type {dialect} is not allowed.")

    execution_info_str = (execution_info or "").strip()
    current_plsql_str = (current_plsql or "").strip()
    language_hint = "PL/pgSQL" if dialect == "postgresql" else "PL/SQL"

    if critical_epoch >= 2 and prior_experiences:
        messages.extend(_build_experience_messages(prior_experiences, execution_info_str))

    error_knowledge_str = _build_error_knowledge(dialect, execution_info_str)
    execution_feedback_block = f"Execution Feedback:\n{execution_info_str or 'N/A'}\n"
    if error_knowledge_str:
        execution_feedback_block += f"Guidance on Error Correction:\n{error_knowledge_str}\n"
    execution_feedback_block += "\n"

    user_content = (
        f"Database Schema:\n{database_schema_str or 'N/A'}\n\n"
        f"{execution_feedback_block}"
        f"Original {language_hint}:\n<start-sql>\n{current_plsql_str}\n<end-sql>\n\n"
        "Instructions:\n"
        "1. Diagnose the failure using the schema and execution feedback.\n"
        "2. Adjust joins, predicates, data transformations, control flow, or exception handling as needed.\n"
        "3. Wrap the previously failing PL/SQL inside <original_plsql>...</original_plsql>.\n"
        "4. Explain the correction strategy, root cause, and validation thoughts inside <correction_experience>...</correction_experience>.\n"
        "5. Present the fully corrected PL/SQL inside <corrected_plsql>...</corrected_plsql>.\n"
        "6. Do not output any other commentary, tags, or metadata."
    )
    messages.append(HumanMessage(content=user_content))

    print("\n" + "=" * 80)
    print("【CORRECTION PROMPT】")
    print("=" * 80)
    print(user_content)
    print("=" * 80 + "\n")
    
    # _print_messages("CORRECTION PROMPT (REQUEST)", messages)

    # 使用超时重试机制调用LLM
    response = _call_review_llm_with_retry(messages, max_retries=3, timeout=120.0)
    content = (response.content or "").strip()

    print("\n" + "=" * 80)
    print("【LLM RESPONSE】")
    print("=" * 80)
    print(content)
    print("=" * 80 + "\n")

    original_match = _ORIGINAL_SECTION_PATTERN.search(content)
    correction_match = _CORRECTION_SECTION_PATTERN.search(content)
    corrected_match = _CORRECTED_SECTION_PATTERN.search(content)

    if not (original_match and correction_match and corrected_match):
        # 缺失 original_section
        if original_match:
            original_section = (original_match.group(1) or "").strip()
            if not original_section:
                original_section = current_plsql_str
        else:
            original_section = current_plsql_str

        # 缺失 correction_section
        if correction_match:
            correction_section = (correction_match.group(1) or "").strip()
            if not correction_section:
                correction_section = "None."
        else:
            correction_section = "None."

        # 缺失 corrected_section
        if corrected_match:
            corrected_section = (corrected_match.group(1) or "").strip()
            if not corrected_section:
                corrected_section = current_plsql_str
        else:
            corrected_section = current_plsql_str

        combined_record = (
            "<original_plsql>\n"
            f"{original_section}\n"
            "</original_plsql>\n"
            "<correction_experience>\n"
            f"{correction_section}\n"
            "</correction_experience>\n"
            "<corrected_plsql>\n"
            f"{corrected_section}\n"
            "</corrected_plsql>"
        )
        return corrected_section, correction_section, combined_record

    original_section = original_match.group(1).strip()
    correction_section = correction_match.group(1).strip()
    corrected_section = corrected_match.group(1).strip()

    combined_record = (
        "<original_plsql>\n"
        f"{original_section}\n"
        "</original_plsql>\n"
        "<correction_experience>\n"
        f"{correction_section}\n"
        "</correction_experience>\n"
        "<corrected_plsql>\n"
        f"{corrected_section}\n"
        "</corrected_plsql>"
    )

    return corrected_section, correction_section, combined_record


def summarize_correction_history(experience_records: Sequence[str]) -> Optional[Dict[str, str]]:
    if not experience_records:
        return None

    parsed_records = []
    for record in experience_records:
        try:
            parsed_records.append(_parse_correction_record(record))
        except ValueError:
            continue

    if not parsed_records:
        return None

    first_original = parsed_records[0][0]
    final_correct = parsed_records[-1][2]
    steps = [entry[1] for entry in parsed_records]

    step_lines = "\n".join(
        f"{idx}. {step}" for idx, step in enumerate(steps, start=1)
    )

    few_shot_examples = (
        'original_plsql: "SELECT * FROM employees",\n'
        'corrected_plsql: "SELECT emp_id, name FROM employees",\n'
        'correction_experience: "Make sure the table and column names match the actual database schema. Do not generate non-existent fields."\n\n'
        'original_plsql: "SELECT * FROM employees",\n'
        'corrected_plsql: "SELECT emp_id, name FROM employees",\n'
        'correction_experience: "Ensure that data types in query conditions and inserted values match the definitions in the table."'
    )

    user_content = (
        "You are transforming accumulated PL/SQL correction attempts into a single generation guideline.\n\n"
        "Reference examples of the desired experience style:\n"
        f"{few_shot_examples}\n\n"
        f"Initial (failing) PL/SQL:\n{first_original}\n\n"
        f"Final Correct PL/SQL:\n{final_correct}\n\n"
        "Correction Attempts:\n"
        f"{step_lines or 'No details available.'}\n\n"
        "Task:\n"
        "- Synthesize the key insights that would allow an engineer to craft the final correct PL/SQL from scratch.\n"
        "- Keep the guidance database-agnostic so it transfers to any dialect or environment.\n"
        "- Avoid referencing concrete table names, column names, variable names, or schema-specific details; use neutral descriptors instead.\n"
        "- Condense the guidance into concise, high-level principles with broad applicability.\n"
        "- Limit the guidance to at most two sentences and include only the most essential, transferable lesson from this successful correction."
    )

    messages = [
        SystemMessage(
            content=(
                "You are an expert PL/SQL reviewer. Summarize lessons from multiple correction attempts "
                "into a reusable, high-level blueprint that applies across databases and scenarios. "
                "Do not mention specific table, column, or variable names. The final guidance must be no more "
                "than two sentences and focus solely on the most impactful, widely applicable insight."
            )
        ),
        HumanMessage(content=user_content),
    ]

    # 使用超时重试机制调用LLM
    response = _call_review_llm_with_retry(messages, max_retries=3, timeout=120.0)
    summary = (response.content or "").strip()
    if not summary:
        return None
    
    print("\n" + "=" * 80)
    print("【LLM SUMMARY】")
    print("=" * 80)
    print(summary)
    print("=" * 80 + "\n")

    return {
        "original_plsql": first_original,
        "corrected_plsql": final_correct,
        "correction_experience": summary,
    }