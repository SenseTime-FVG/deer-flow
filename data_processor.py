import asyncio
import json
from main import ask
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def process_single_query(
    query_data,
    output_path,
    debug,
    max_plan_iterations,
    max_step_num,
    enable_background_investigation,
    semaphore, # 新增信号量参数
):
    """处理单个查询的异步函数，并使用信号量控制并发"""
    async with semaphore: # 进入信号量，当达到并发上限时会等待
        query = query_data.get("query")
        file_path = query_data.get("file_path")

        if not query:
            logger.warning("Warning: Query data is missing 'query' field. Skipping.")
            return

        files_to_process = [file_path] if isinstance(file_path, str) else file_path if file_path else []

        logger.info(f"Processing Query: '{query}', Files: {files_to_process}")
        await ask( # 直接 await ask，因为我们在这里处理单个任务
            question=query,
            files=files_to_process,
            debug=debug,
            max_plan_iterations=max_plan_iterations,
            max_step_num=max_step_num,
            enable_background_investigation=enable_background_investigation,
            output_path=output_path
        )
        logger.info(f"Finished Query: '{query}'")

async def process_query_from_jsonl_limited_concurrency(
    jsonl_file_path: str,
    output_path: str,
    debug: bool = False,
    max_plan_iterations: int = 1,
    max_step_num: int = 3,
    enable_background_investigation: bool = True,
    concurrency_limit: int = 5, # 新增并发限制参数
):
    """
    Reads queries and file paths from a JSONL file and processes them with a limited concurrency.
    """
    if not os.path.exists(jsonl_file_path):
        logger.error(f"Error: JSONL file not found at '{jsonl_file_path}'")
        return

    # --- 断点续传功能：读取已处理的查询 ---
    processed_queries = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        output_data = json.loads(line.strip())
                        # 检查 'messages' 是否存在且至少有一个元素包含 'content'
                        if (
                            isinstance(output_data, dict)
                            and "messages" in output_data
                            and isinstance(output_data["messages"], list)
                            and len(output_data["messages"]) > 0
                            and "content" in output_data["messages"][0]
                        ):
                            processed_queries.add(output_data["messages"][0]["content"])
                        else:
                            logger.warning(f"跳过 '{output_path}' 中第 {line_num} 行格式不正确的输出。")
                    except json.JSONDecodeError:
                        logger.warning(f"跳过 '{output_path}' 中第 {line_num} 行无效的 JSON。")
        except Exception as e:
            logger.error(f"读取输出 JSONL 文件 '{output_path}' 时发生错误: {e}")
    logger.info(f"在 '{output_path}' 中找到 {len(processed_queries)} 个先前已处理的查询。")
    # --- 断点续传功能结束 ---
    
    
    # 创建一个信号量，限制同时运行的任务数量
    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = []

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                query = data.get("query")
                
                if not query:
                    logger.warning(f"警告: 第 {line_num} 行的查询数据缺少 'query' 字段。跳过。")
                    continue
                
                # --- 断点续传检查 ---
                if query in processed_queries:
                    logger.info(f"跳过第 {line_num} 行已处理的查询: '{query}'")
                    continue
                # --- 断点续传检查结束 ---
                
                tasks.append(
                    process_single_query(
                        data,
                        output_path,
                        debug,
                        max_plan_iterations,
                        max_step_num,
                        enable_background_investigation,
                        semaphore,
                    )
                )
            except json.JSONDecodeError:
                logger.error(
                    f"Error: Invalid JSON on line {line_num} in '{jsonl_file_path}'. Skipping."
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred on line {line_num}: {e}"
                )

    if tasks:
        logger.info(f"Processing {len(tasks)} queries with a concurrency limit of {concurrency_limit}...")
        await asyncio.gather(*tasks) # 仍然使用 gather，但每个任务都会先尝试获取信号量
        logger.info("All queries processed.")
    else:
        logger.info("No valid queries found in the JSONL file to process.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process queries from a JSONL file concurrently."
    )
    parser.add_argument(
        "--jsonl_file", type=str, help="Path to the JSONL file containing queries."
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to the history store"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--max_plan_iterations",
        type=int,
        default=1,
        help="Maximum number of plan iterations (default: 1).",
    )
    parser.add_argument(
        "--max_step_num",
        type=int,
        default=3,
        help="Maximum number of steps in a plan (default: 3).",
    )
    parser.add_argument(
        "--no-background-investigation",
        action="store_false",
        dest="enable_background_investigation",
        help="Disable background investigation before planning.",
    )
    parser.add_argument(
        "--concurrency_limit",
        type=int,
        default=5, # 默认并发限制为 5
        help="Maximum number of concurrent tasks (default: 5).",
    )

    args = parser.parse_args()

    asyncio.run(
        process_query_from_jsonl_limited_concurrency(
            jsonl_file_path=args.jsonl_file,
            output_path=args.output_path,
            debug=args.debug,
            max_plan_iterations=args.max_plan_iterations,
            max_step_num=args.max_step_num,
            enable_background_investigation=args.enable_background_investigation,
            concurrency_limit=args.concurrency_limit,
        )
    )