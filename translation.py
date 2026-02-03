from langgraph.graph import StateGraph, START, END
from state.translation_state import TranslationState
from agent.translation.translation_node import summary, style_transfer
def entry_node(state: TranslationState) -> TranslationState:
    state["epoch"] = 0
    state["dialect"] = "postgresql"
    state["plsql_collection"] = ["1", "2", "3"]
    return state

def management_node(state: TranslationState) -> TranslationState:
    state["epoch"] += 1
    return state

def route_on_whether_continue_translation(state: TranslationState) -> TranslationState:
    if state["epoch"] >= len(state["plsql_collection"]):
        return "end_node"
    else:
        return "summary_node"

def summary_node(state: TranslationState) -> TranslationState:
    state = summary(state)
    return state

def style_node(state: TranslationState) -> TranslationState:
    state = style_transfer(state)
    return state

def init_translation_graph():
    translation_graph = StateGraph(TranslationState)

    translation_graph.add_node("entry_node", entry_node)
    translation_graph.add_node("management_node", management_node)
    translation_graph.add_node("summary_node", summary_node)
    translation_graph.add_node("style_node", style_node)

    translation_graph.add_edge(START, "entry_node")
    translation_graph.add_edge("entry_node", "management_node")
    translation_graph.add_conditional_edges(
        "management_node",
        route_on_whether_continue_translation,
        {
            "summary_node": "summary_node",
            "end_node": END
        }
    )
    translation_graph.add_edge("summary_node", "style_node")
    translation_graph.add_edge("style_node", "management_node")

    return translation_graph.compile()