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


IR_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a meticulous assistant who produces exhaustive intermediate representations (IRs) for PL/SQL programs. "
            "Always follow the user's formatting and stylistic requirements precisely."
        ),
        (
            "user",
            (
                "You are an expert in {dialect} database and PL/SQL programming. Study the existing IR and PL/SQL pair below. "
                "You must keep the same database context (database name: {database_name}) and the same tables:\n"
                "{tables_formatted}\n\n"
                "Generate EXACTLY five new IR descriptions that comply with ALL of the following IR authoring requirements:\n"
                "1. The description must be EXTREMELY DETAILED and comprehensive\n"
                "2. Include ALL parameters with their types and purposes\n"
                "3. Describe EVERY operation performed (updates, inserts, deletes, selects, etc.)\n"
                "4. Mention ALL table names, column names, and conditions used\n"
                "5. Explain any function calls or special operations (like UPPER(), LOWER(), etc.)\n"
                "6. Describe the logic flow and any conditional statements\n"
                "7. Use clear, precise technical language\n"
                "8. Start with the type of database object (procedure, function, trigger)\n"
                "9. The description must start with the word 'Write' followed by a natural-language explanation of what the PL/SQL code does, phrased as an instruction or specification.\n"
                "10. Do not include any mention of transactions, commits, rollbacks, locks, concurrency, isolation levels, or any database engine internals.\n"
                "11. Do not include any concluding or summarizing sentences. End the output immediately after the full logical description of the code.\n"
                "12. Preserve the PL/SQL unit kind implied by the original IR/PLSQL pair: if the original corresponds to a FUNCTION, each generated IR must specify a FUNCTION; if PROCEDURE, specify a PROCEDURE; if TRIGGER, specify a TRIGGER.\n"
                "13. Keep the complexity of each generated IR approximately the same as the original IR, including a similar number of described logical statements and a comparable expected PL/SQL statement count.\n\n"
                "The new IRs must stay within the same narrative universe as the original IR but explore different semantic goals or business questions. "
                "Focus on semantic variety rather than superficial wording differences.\n\n"
                "Original IR:\n{original_ir}\n\n"
                "Original PL/SQL:\n```sql\n{original_plsql}\n```\n\n"
                "Output format: Return a JSON array of five objects. Each object MUST contain exactly one key \"ir\" whose value is the IR text. "
                "Do not include any additional keys or commentary."
            )
        ),
    ]
)

PLSQL_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior PL/SQL engineer. Produce syntactically correct and semantically precise PL/SQL programs that align exactly with the provided IR specifications."
        ),
        (
            "user",
            (
                "You are an expert in {dialect} database and PL/SQL programming. "
                "Using the database name {database_name} and the tables listed below:\n"
                "{tables_formatted}\n\n"
                "You will be given five IR descriptions (Intermediate Representations). For each IR, generate a distinct PL/SQL program that fulfills the IR exactly, "
                "using the same database context. Maintain a 1-to-1 order correspondence between the IRs and the generated PL/SQL programs.\n\n"
                "Priorities:\n"
                "1. Ensure every program aligns perfectly with its IR specification.\n"
                "2. Emphasize structural differentiation across the five programs (e.g., reorganize loops, reorder conditional branches, introduce/refactor subqueries, "
                "split or combine statements) while keeping semantics consistent with the IRs.\n"
                "3. Reuse the same tables and relevant columns, respecting the IR guidance.\n"
                "4. Do NOT add comments. Focus on business logic; omit explicit transaction handling.\n"
                "5. Each program must be valid {dialect} PL/SQL.\n\n"
                "IRs (in order):\n{irs_serialized}\n\n"
                "Output format: Return a JSON array of five objects. Each object must contain exactly one key \"plsql\" whose value is the full PL/SQL code string. "
                "Provide no additional text or keys."
            )
        ),
    ]
)


def ir_expansion_agent(dialect: str, selected_seed: List[Dict]) -> List[Dict]:
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

            original_ir = seed["ir"]
            original_plsql = seed["plsql"]
            database_name = seed["database_name"]
            tables = seed["tables"]

            tables_formatted = "\n".join(f"- {table_name}" for table_name in tables) if tables else "- (none provided)"

            ir_messages = IR_PROMPT_TEMPLATE.format_messages(
                dialect=dialect,
                database_name=database_name,
                tables_formatted=tables_formatted,
                original_ir=original_ir,
                original_plsql=original_plsql,
            )
            
            print("\n" + "=" * 80)
            print("【IR EXPANSION PROMPT】")
            print("=" * 80)
            print(ir_messages[0].content)
            print("=" * 80 + "\n")
            
            ir_response = _call_llm_with_retry(ir_messages)
            ir_content = ir_response.content.strip()
            ir_items_raw = _load_json_array(ir_content)
            
            print("\n" + "=" * 80)
            print("【IR EXPANSION RESPONSE】")
            print("=" * 80)
            print(ir_content)
            print("=" * 80 + "\n")

            if ir_items_raw is None:
                print(f"【WARNING】Seed index {seed_index}: Failed to parse IR response JSON. Skipping this seed.")
                continue

            if len(ir_items_raw) != 5:
                print(f"【WARNING】Seed index {seed_index}: expected 5 IR items but received {len(ir_items_raw)}. Using available items.")
                if len(ir_items_raw) == 0:
                    print(f"【WARNING】Seed index {seed_index}: No IR items received. Skipping this seed.")
                    continue

            ir_texts = []
            for idx, item in enumerate(ir_items_raw):
                if not isinstance(item, dict) or "ir" not in item:
                    print(f"【WARNING】Seed index {seed_index}: IR item {idx} is invalid: {item}. Skipping this item.")
                    continue
                ir_texts.append(item["ir"].strip())

            if len(ir_texts) == 0:
                print(f"【WARNING】Seed index {seed_index}: No valid IR texts extracted. Skipping this seed.")
                continue

            ir_payload_serialized = json.dumps(ir_texts, ensure_ascii=False, indent=2)

            plsql_messages = PLSQL_PROMPT_TEMPLATE.format_messages(
                dialect=dialect,
                database_name=database_name,
                tables_formatted=tables_formatted,
                irs_serialized=ir_payload_serialized,
            )
            
            print("\n" + "=" * 80)
            print("【PLSQL GENERATION PROMPT】")
            print("=" * 80)
            print(plsql_messages[0].content)
            print("=" * 80 + "\n")
            
            plsql_response = _call_llm_with_retry(plsql_messages)
            plsql_content = plsql_response.content.strip()
            plsql_items_raw = _load_json_array(plsql_content)
            
            print("\n" + "=" * 80)
            print("【LLM RESPONSE】")
            print("=" * 80)
            print(plsql_content)
            print("=" * 80 + "\n")

            if plsql_items_raw is None:
                print(f"【WARNING】Seed index {seed_index}: Failed to parse PL/SQL response JSON. Skipping this seed.")
                continue

            if len(plsql_items_raw) != len(ir_texts):
                print(f"【WARNING】Seed index {seed_index}: expected {len(ir_texts)} PL/SQL items but received {len(plsql_items_raw)}. Using available items.")
                if len(plsql_items_raw) == 0:
                    print(f"【WARNING】Seed index {seed_index}: No PL/SQL items received. Skipping this seed.")
                    continue

            plsql_texts = []
            for idx, item in enumerate(plsql_items_raw):
                if not isinstance(item, dict) or "plsql" not in item:
                    print(f"【WARNING】Seed index {seed_index}: PL/SQL item {idx} is invalid: {item}. Skipping this item.")
                    continue
                plsql_texts.append(item["plsql"].strip())

            if len(plsql_texts) == 0:
                print(f"【WARNING】Seed index {seed_index}: No valid PL/SQL texts extracted. Skipping this seed.")
                continue

            # 只处理成对的 IR 和 PL/SQL
            min_len = min(len(ir_texts), len(plsql_texts))
            for i in range(min_len):
                expansions.append(
                    {
                        "ir": ir_texts[i],
                        "plsql": plsql_texts[i],
                        "database_name": database_name,
                        "tables": list(tables),
                    }
                )
            
            print(f"【INFO】Seed index {seed_index}: Successfully processed {min_len} expansion(s).")
            
        except Exception as e:
            print(f"【ERROR】Seed index {seed_index}: Unexpected error occurred: {type(e).__name__}: {str(e)}")
            print(f"【ERROR】Traceback: {repr(e)}")
            print(f"【WARNING】Skipping seed index {seed_index} due to error.")
            continue

    print(f"\n【INFO】Total expansions generated: {len(expansions)}")
    return expansions