import argparse
from seed_generation import init_seed_generation_graph
from state.seed_generation_state import SeedGenerationState

def save_graph_image(graph, output_path="graph.png"):
    """Save the graph as a PNG image file"""
    png_data = graph.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_data)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Seed Generation Script")
    
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch number (default: 0)"
    )
    
    parser.add_argument(
        "--target-plsql-number",
        type=int,
        default=6,
        help="Target PLSQL number (default: 6)"
    )
    
    parser.add_argument(
        "--current-plsql-number",
        type=int,
        default=0,
        help="Current PLSQL number (default: 0)"
    )
    
    parser.add_argument(
        "--dialect",
        type=str,
        default="postgresql",
        choices=["postgresql", "oracle"],
        help="SQL dialect (default: postgresql)"
    )
    
    parser.add_argument(
        "--need-correction",
        action="store_true",
        help="Whether correction is needed (default: False)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 初始化图
    seed_generation_graph = init_seed_generation_graph()
    save_graph_image(seed_generation_graph, "./image/graph/seed_generation_graph.png")

    # 使用命令行参数创建 state
    state = SeedGenerationState(
        epoch=args.epoch,
        target_plsql_number=args.target_plsql_number,
        current_plsql_number=args.current_plsql_number,
        dialect=args.dialect,
        need_correction=args.need_correction,
        stage="generation"
    )

    print(state)
    result = seed_generation_graph.invoke(
        state, 
        config={"recursion_limit": 999999999}  # 将限制改为 100，或其他你需要的值
    )
    print(result)