import re
import random
from typing import List
from langchain.prompts import ChatPromptTemplate

from state.translation_state import TranslationState
from config.common import weak_llm_config
from util.llm_util import call_llm_with_retry


style_options = [
    "Imperative: A direct command or instruction. Often starts with a verb. (e.g., 'List...', 'Show...', 'Find...')",
    "Formal: Uses proper, structured, and polite language, adhering to grammatical standards. Common in professional or academic writing.",
    "Descriptive: Elaborates with adjectives, adverbs, and clauses to provide rich detail, potentially making the sentence longer.",
    "Colloquial: Employs informal, everyday language, including slang, idioms, and contractions, as used in casual speech.",
    "Vague: Uses imprecise terms and subjective qualifiers, requiring interpretation of fuzzy concepts. (e.g., 'some recent good-selling products')",
    "Interrogative: Poses a direct question. This is the most common style for queries. (e.g., 'What are...', 'How many...', 'Which product...')",
    "Concise: Extremely brief and to the point, using the fewest words possible to convey the core request without elaboration.",
]

summarize_prompt = ChatPromptTemplate(
    [
        (
            "user",
            "You are an expert in PLSQL programming and natural language processing."
            "Below are some complex natural language description (ir) related to database operations. Please rewrite and summarize them into concise, clear natural language that can be directly used to generate corresponding PLSQL code.\n"
            "### Requirements:\n"
            "1. Extract the core operational intent from the complex description\n"
            "2. Maintain technical accuracy for PLSQL generation\n"
            "3. If the original text contains semantic information about specific tasks or output formats, this information should be preserved.\n"
            "### Complex Natural Language Input:\n {ir_description} \n"
            "### Output Format:\n"
            "IMPORTANT: Output ONLY the natural language in the following format, WITHOUT any additional explanations, descriptions, or extra text.\n"
            "Each generated query must be wrapped in <start-nl> and <end-nl> tags:\n\n"
            "<start-nl>\n"
            "[Natural Language1 here]\n"
            "<end-nl>\n\n"
            "<start-nl>\n"
            "[Natural Language2 here]\n"
            "<end-nl>\n\n"
            "### Example 1:\n"
            "<start-nl>\n"
            "Write a PL/SQL block that calculates the total sales for a specific product from the orders table. If the product's total sales exceed a threshold value (e.g., 10,000), display a message indicating that the product is a bestseller. Otherwise, display a message indicating that it needs more promotion.\n"
            "<end-nl>\n"
            "### Example 2:\n"
            "<start-nl>\n"
            "Create a PL/SQL procedure that calculates the total salary of all employees in the SALES department. If the department does not exist, raise an application error. Return the total salary via an OUT parameter. Log the execution time into a table named QUERY_LOG with columns (query_name, elapsed_ms).\n"
            "<end-nl>\n\n"
            "Please only return the rewritten concise natural language description, without any additional explanations or code examples.",
        )
    ]
)


rewrite_prompt = ChatPromptTemplate(
    [
        (
            "user",
            "You are an expert in Text-to-PLSQL conversion and natural language rewriting.\n"
            "Your task is to rewrite a given natural language query into a ** assigned style** to create alternative training data or test cases.\n"
            "### Style Definition:\n"
            "{random_style}\n"
            "### Original Natural Language Query:\n"
            "{nl_query}\n"
            "### Instructions:\n"
            "1. **CRITICAL:** The core semantic meaning and the resulting SQL logic must remain **unchanged**.\n"
            "2. Make the rewritten query sound natural and fluent for the assigned style.\n"
            "### Output Format:\n"
            "IMPORTANT: Output ONLY the rewritten natural language query in the following format, WITHOUT any additional explanations, descriptions, or extra text.\n"
            "The output must be wrapped in <start-nl> and <end-nl> tags:\n\n"
            "<start-nl>\n"
            "[Your rewritten natural language query here]\n"
            "<end-nl>",
        )
    ]
)


def _call_translation_llm_with_retry(
    prompt, max_retries: int = 3, timeout: float = 120.0
):
    """
    使用超时和重试机制调用摘要LLM

    Args:
        prompt: 要发送给LLM的prompt
        max_retries: 最大重试次数，默认3次
        timeout: 每次调用的超时时间（秒），默认120秒

    Returns:
        LLM响应对象
    """
    ir_model_cfg = weak_llm_config.get("translation_model", {})

    def llm_call(llm):
        """LLM调用函数"""
        return llm.invoke(prompt)

    # 使用超时重试机制调用
    response = call_llm_with_retry(
        llm_func_factory=llm_call,
        model_config=ir_model_cfg,
        max_retries=max_retries,
        timeout=timeout,
        model_name="translation_model",
        verbose=True,
    )

    return response


def summary(state: TranslationState):
    ir_text = ""
    natural_languages = []
    for ir in state["ir_description"]:
        ir_text += ir + "\n" + "------" + "\n"

    response = _call_translation_llm_with_retry(
        summarize_prompt.format(ir_description=ir_text)
    )

    translated_text = response.content.strip()
    pattern = r"<start-nl>\n(.*?)\n<end-nl>"
    matches = re.findall(pattern, translated_text, re.DOTALL)

    for i, match in enumerate(matches, 1):
        natural_languages.append(match.strip())

    state["natural_languages"] = natural_languages
    return state


def style_transfer(state: TranslationState):
    state["rewrite_description"] = []
    for i, nl in enumerate(state["natural_languages"]):
        random_style = random.choice(style_options)
        response = _call_translation_llm_with_retry(
            rewrite_prompt.format(random_style=random_style, nl_query=nl)
        )
        translated_text = response.content.strip()
        pattern = r"<start-nl>\n(.*?)\n<end-nl>"
        matches = re.findall(pattern, translated_text, re.DOTALL)
        state["rewrite_description"].append(matches[0].strip())
    return state


def summary_list(ir_description: List[str]):
    ir_text = ""
    natural_languages = []
    for ir in ir_description:
        ir_text += ir + "\n" + "------" + "\n"

    response = _call_translation_llm_with_retry(
        summarize_prompt.format(ir_description=ir_text)
    )

    translated_text = response.content.strip()
    pattern = r"<start-nl>\n(.*?)\n<end-nl>"
    matches = re.findall(pattern, translated_text, re.DOTALL)

    for i, match in enumerate(matches, 1):
        natural_languages.append(match.strip())

    return natural_languages


def style_transfer_list(natural_languages: List[str]):
    rewrite_description = []
    for i, nl in enumerate(natural_languages):
        random_style = random.choice(style_options)
        response = _call_translation_llm_with_retry(
            rewrite_prompt.format(random_style=random_style, nl_query=nl)
        )
        translated_text = response.content.strip()
        pattern = r"<start-nl>\n(.*?)\n<end-nl>"
        matches = re.findall(pattern, translated_text, re.DOTALL)
        rewrite_description.append(matches[0].strip())
    return rewrite_description
