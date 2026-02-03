from translation import init_translation_graph
from state.translation_state import TranslationState

def save_graph_image(graph, output_path="graph.png"):
    """Save the graph as a PNG image file"""
    png_data = graph.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_data)

if __name__ == "__main__":
    translation_graph = init_translation_graph()
    save_graph_image(translation_graph, "./image/graph/translation_graph.png")

    state = TranslationState(
        epoch=0,
        dialect="postgresql",
        plsql_collection=["1", "2", "3"]
    )

    print(state)
    result = translation_graph.invoke(state)
    print(result)