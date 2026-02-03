import json
import re
from typing import List, Dict

from langchain.prompts import ChatPromptTemplate

from config.common import weak_llm_config
from util.llm_util import call_llm_with_retry


def _call_llm_with_retry(prompt_messages, max_retries: int = 3, timeout: float = 120.0):
    model_cfg = weak_llm_config.get("expansion_model", {})

    def llm_call(llm):
        return llm.invoke(prompt_messages)

    response = call_llm_with_retry(
        llm_func_factory=llm_call,
        model_config=model_cfg,
        max_retries=max_retries,
        timeout=timeout,
        model_name="expansion_model",
        verbose=True
    )
    return response


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def _load_json_array(content: str):
    cleaned = _strip_code_fences(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                print(f"【ERROR】Failed to parse JSON from LLM response, returning None")
                print(f"Content preview: {content[:500]}...")
                return None
        print(f"【ERROR】Failed to parse JSON from LLM response, returning None")
        print(f"Content preview: {content[:500]}...")
        return None


PLSQL_VARIATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a veteran PL/SQL optimizer who can reimagine code structures without altering semantics. "
            "Ensure generated programs strictly obey the provided constraints."
        ),
        (
            "user",
            (
                "You are an expert in {dialect} database and PL/SQL programming. "
                "Given the database context (database name: {database_name}) and the tables:\n"
                "{tables_formatted}\n\n"
                "You will receive an IR (Intermediate Representation) and an existing PL/SQL implementation. "
                "Produce EXACTLY five new PL/SQL programs that remain faithful to the IR while showcasing substantial structural diversity.\n\n"
                "Goals for structural diversity include (but are not limited to):\n"
                "- Reordering logically independent operations where appropriate\n"
                "- Transforming sequential logic into nested structures or vice versa\n"
                "- Replacing equivalent operators or control constructs that preserve semantics\n"
                "- Introducing/merging subqueries, cursors, or loop constructs while keeping behavior consistent\n"
                "- Adjusting control-flow granularity (e.g., splitting combined conditions, consolidating multiple checks)\n\n"
                "Constraints:\n"
                "1. Honor every detail in the IR; the semantics must remain identical.\n"
                "2. Maintain compatibility with the specified {dialect} dialect.\n"
                "3. Do NOT introduce comments or transaction control statements.\n"
                "4. Use only the provided database and tables.\n"
                "5. Ensure the five programs differ meaningfully in structure, not just in superficial wording.\n"
                "6. Preserve the PL/SQL unit kind: if the original is a FUNCTION, each generated one must be a FUNCTION; if PROCEDURE, each must be a PROCEDURE; if TRIGGER, each must be a TRIGGER.\n"
                "7. Keep the number of statements in each generated PL/SQL program approximately the same as in the original.\n\n"
                "IR:\n{ir_text}\n\n"
                "Existing PL/SQL Implementation:\n```sql\n{original_plsql}\n```\n\n"
                "Output format: Return a JSON array of five objects. Each object must contain exactly one key \"plsql\" whose value is the full PL/SQL code string. "
                "Provide no additional text or fields."
            )
        ),
    ]
)


def plsql_expansion_agent(dialect: str, selected_seed: List[Dict]) -> List[Dict]:
    if not selected_seed:
        print("【WARNING】selected_seed is empty, returning empty list.")
        return []

    expansions: List[Dict] = []

    for seed_index, seed in enumerate(selected_seed):
        try:
            # 检查必需的键
            missing_keys = [key for key in ("ir", "plsql", "database_name", "tables") if key not in seed]
            if missing_keys:
                print(f"【WARNING】Seed index {seed_index} is missing required keys: {missing_keys}. Skipping this seed.")
                continue

            ir_text = seed["ir"]
            original_plsql = seed["plsql"]
            database_name = seed["database_name"]
            tables = seed["tables"]

            tables_formatted = "\n".join(f"- {table_name}" for table_name in tables) if tables else "- (none provided)"

            prompt_messages = PLSQL_VARIATION_PROMPT_TEMPLATE.format_messages(
                dialect=dialect,
                database_name=database_name,
                tables_formatted=tables_formatted,
                ir_text=ir_text,
                original_plsql=original_plsql,
            )
            
            print("\n" + "=" * 80)
            print("【PLSQL EXPANSION PROMPT】")
            print("=" * 80)
            print(prompt_messages[0].content)
            print("=" * 80 + "\n")

            response = _call_llm_with_retry(prompt_messages)
            response_content = response.content.strip()
            plsql_items_raw = _load_json_array(response_content)
            
            print("\n" + "=" * 80)
            print("【PLSQL EXPANSION RESPONSE】")
            print("=" * 80)
            print(response_content)
            print("=" * 80 + "\n")

            if plsql_items_raw is None:
                print(f"【WARNING】Seed index {seed_index}: Failed to parse PL/SQL response JSON. Skipping this seed.")
                continue

            if len(plsql_items_raw) != 5:
                print(f"【WARNING】Seed index {seed_index}: expected 5 PL/SQL variants but received {len(plsql_items_raw)}. Using available items.")
                if len(plsql_items_raw) == 0:
                    print(f"【WARNING】Seed index {seed_index}: No PL/SQL items received. Skipping this seed.")
                    continue

            valid_count = 0
            for idx, item in enumerate(plsql_items_raw):
                if not isinstance(item, dict) or "plsql" not in item:
                    print(f"【WARNING】Seed index {seed_index}: PL/SQL item {idx} is invalid: {item}. Skipping this item.")
                    continue
                expansions.append(
                    {
                        "ir": ir_text,
                        "plsql": item["plsql"].strip(),
                        "database_name": database_name,
                        "tables": list(tables),
                    }
                )
                valid_count += 1
            
            if valid_count == 0:
                print(f"【WARNING】Seed index {seed_index}: No valid PL/SQL items extracted. Skipping this seed.")
            else:
                print(f"【INFO】Seed index {seed_index}: Successfully processed {valid_count} expansion(s).")
            
        except Exception as e:
            print(f"【ERROR】Seed index {seed_index}: Unexpected error occurred: {type(e).__name__}: {str(e)}")
            print(f"【ERROR】Traceback: {repr(e)}")
            print(f"【WARNING】Skipping seed index {seed_index} due to error.")
            continue

    print(f"\n【INFO】Total expansions generated: {len(expansions)}")
    return expansions