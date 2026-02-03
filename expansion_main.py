import argparse
from expansion import init_expansion_graph
from state.expansion_state import ExpansionState


# def save_graph_image(graph, output_path="graph.png") -> None:
#     png_data = graph.get_graph().draw_mermaid_png()
#     with open(output_path, "wb") as f:
#         f.write(png_data)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Expansion Script")
    
    parser.add_argument(
        "--target-plsql-number",
        type=int,
        default=20,
        help="Target PLSQL number (default: 20)"
    )
    
    parser.add_argument(
        "--dialect",
        type=str,
        default="postgresql",
        choices=["postgresql", "oracle"],
        help="SQL dialect (default: postgresql)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 初始化图
    expansion_graph = init_expansion_graph()
    # save_graph_image(expansion_graph, "./image/graph/expansion_graph.png")

    # 使用命令行参数创建 state
    state = ExpansionState(
        target_plsql_number=args.target_plsql_number,
        dialect=args.dialect
    )

    print(state)
    result = expansion_graph.invoke(
        state, 
        config={"recursion_limit": 999999999}
    )
    print(result)