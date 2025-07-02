# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import copy
import logging
import os
import os.path as osp
import datetime
from src.graph import build_graph, build_graph_with_memory, build_searcher_subgraph_with_memory
from src.utils.file_descriptors import file2resource, resources2user_input
import uuid
import shutil
from langgraph.types import Command
import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.graph.nodes import SearcherNode
from src.graph.tools.tool_manager import ToolManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
# graph = build_graph()


def get_init_state(
        user_input: str | list[dict], 
        is_select_searcher: bool,
        is_break_for_plan: bool,
        enable_background_investigation: bool) -> str | list[dict]:
    """
    1. 对用户输入进行预处理以初始化状态。
    2. 创建会话目录
    3. 将用户输入文件转换为资源列表。
    参数：
        user_input：用户的查询或请求，类型为字符串或字典列表
        enable_background_investigation：若为 True，则在规划前进行网络搜索以增强上下文信息
    返回值：
        经过预处理的用户输入
        用户输入可以是字符串，也可以是字典列表，每个字典包含一个 'type' 键以及 'text' 或 'file_url' 键。
    """
    
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4().hex[:8])
    session_dir = osp.join(os.environ.get('SESSION_DIR', './sessions'), session_id)
    if not osp.exists(session_dir):
        os.makedirs(session_dir)

    if isinstance(user_input, str):
        user_input_text = user_input
        resources = []
    elif isinstance(user_input, list):
        if len(user_input) == 0:
            raise ValueError("User input cannot be empty")
        
        user_input_text = '\n\n'.join([f['text'] for f in user_input if f['type'] == 'text'])
        
        for content in user_input:
            # copy file from file_url and save to session_dir
            if content['type'] == 'file_url':

                file_url = content['file_url']['url']
                file_name = osp.basename(file_url)
                file_path = osp.join(session_dir, file_name)
                if osp.exists(file_path):
                    # 在文件名stem上增加一个后缀, ...
                    file_name_stem = osp.splitext(file_name)[0]
                    file_name = file_name_stem + '_' + str(uuid.uuid4().hex[:8]) + osp.splitext(file_name)[1]
                    file_path = osp.join(session_dir, file_name)
                shutil.copy(file_url, file_path)
                content['file_url']['url'] = file_path
                    
        files = [f for f in user_input if f['type'] == 'file_url']
        resources = [file2resource(f['file_url']['url']) for f in files]
        
        for res_i, resource in enumerate(resources):
            resources[res_i]['resource_id'] = res_i
        user_input_text = user_input_text + "\n\n" + resources2user_input(resources)

    else:
        raise ValueError("Invalid user input type")
    
    return {
        "messages": [{"role": "user", "content": user_input_text}],
        "resources": resources,
        "locale": "zh-CN",
        "auto_accepted_plan": True,
        "enable_background_investigation": enable_background_investigation,
        "session_id": session_id,
        "session_dir": session_dir,

        "supervisor_iterate_time":0,
        "tool_call_iterate_time":0,
        "history_clear": False,
        
        # 是否需要中断进行planner数据收集
        "is_break": is_break_for_plan,
        "planner_node_capture": False,
        
        # 是否收集searcher数据
        "is_select_searcher": is_select_searcher,
        
        # 是否使用llm模拟用户回答
        "use_llm_simulate_user": True
    }


async def run_agent_workflow_async(
    user_input: str | list[dict],
    debug: bool = False,
    max_plan_iterations: int=1,
    max_step_num: int=3,
    enable_background_investigation: bool = True,
    output_path: str = None,
    is_select_searcher: bool=False,
    is_break_for_plan: bool=False
):
    """Run the agent workflow asynchronously with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations
        max_step_num: Maximum number of steps in a plan
        enable_background_investigation: If True, performs web search before planning to enhance context

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"Starting async workflow with user input: {user_input}")
    
    
    
    initial_state = get_init_state(user_input, is_select_searcher, is_break_for_plan, enable_background_investigation)

    config = {
        "configurable": {
            "thread_id": "default",
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "max_search_results": 5,
            "max_toolcall_iterate_times": 5,
            "max_supervisor_iterate_times": 5,
            "mcp_settings": {
                "servers": {
                    "doc_parser": {
                        "transport": "sse",
                        "url": "http://127.0.0.1:8010/sse",
                        "enabled_tools": ["parse_doc"],
                        "add_to_agents": ["analyzer"],
                    },
                    "Sandbox": {
                        "transport": "sse",
                        "url": "http://0.0.0.0:8015/sse",
                        "enabled_tools": ["run_code_sandbox_fusion"],
                        "add_to_agents": ["coder"],
                    },
                }
            },
        },
        "recursion_limit": 50, #为整个的调度次数
    }
    
    # 收集searcher数据限定在searcher的子图
    if initial_state.get("is_select_searcher"):
        searcher_graph = build_searcher_subgraph_with_memory()
        async for s in searcher_graph.astream(
                input=initial_state, config=config, stream_mode="values"
            ):
            logger.info(f"searcher 步骤状态: {s}")
        
        serialize_searcher_step = serialize_step(s)
        
        with open(output_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(serialize_searcher_step, ensure_ascii=False)+"\n")
        return
    
    
    should_stop_workflow = False
    
    graph = build_graph_with_memory()
    while True:
        async for s in graph.astream(
            input=initial_state, config=config, stream_mode="values"
        ):
            if isinstance(s, dict) and s.get("planner_node_capture"):
                # 遍历 messages 列表，将每个 Message 转换为字典
                serialize_plan_step = serialize_step(s)
                
                # 外重循环终止标志
                should_stop_workflow = True
                
                if output_path:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(serialize_plan_step, ensure_ascii=False)+"\n")
                    logger.info(f"Workflow state saved to {output_path}")
                break
                
            if "final_report" in s:
                logger.info(f"Final result:\n{s['final_report']}")
                should_stop_workflow = True
                break
            
        
            # if isinstance(s, dict) and "messages" in s:
            #     # 默认会继承全部历史记录，这里如果设置了clear则只保留当前对话
            #     if s["history_clear"]:
            #         s["messages"] = s["messages"][-1]
            #         s["history_clear"] = False
            #         print("*" * 50)
            #         print(s["messages"])
            #         print("*" * 50)
                    
        if isinstance(s, dict) and "__interrupt__" in s:
            # print(f"Interrupt: {s['__interrupt__']}")
            feedback = input(s['__interrupt__'][0].value + ": ")
            initial_state = Command(resume=feedback)

        if should_stop_workflow:
            logger.info("Workflow stopped as requested after plan generation or return final.")
            return 
        
        logger.info("Async workflow completed successfully")
            

def serialize_step(s):
    
    serializable_messages = []
    for msg in s.get('messages'):
        if isinstance(msg, HumanMessage):
            serializable_messages.append({
                "type": "human", # 明确消息类型
                "content": msg.content,
                "additional_kwargs": msg.additional_kwargs,
                "response_metadata": msg.response_metadata,
                "id": msg.id,
                "name":msg.name
                # 根据需要添加其他属性
            })
        # 如果还有其他类型的消息（如 AIMessage），也需要进行相应的处理
        elif isinstance(msg, AIMessage):
            serializable_messages.append({
                "type": "assistant",
                "content": msg.content,
                "additional_kwargs": msg.additional_kwargs,
                "response_metadata": msg.response_metadata,
                "id": msg.id,
                "name":msg.name
            })
        elif isinstance(msg, ToolMessage):
            serializable_messages.append({
                "type": "tool",
                "content": msg.content,
                "additional_kwargs": msg.additional_kwargs,
                "response_metadata": msg.response_metadata,
                "id": msg.id,
                "name":msg.name
            })
        else:
            # 如果有其他非 HumanMessage 且非 AIMessage 的消息类型，直接保留或报错
            serializable_messages.append(msg) # 或者选择跳过，或者报错

    # 创建一个可序列化版本的 state
    serializable_state = s.copy()
    serializable_state['messages'] = serializable_messages
        # print(s_as_dict.type)
    # print(f"plan返回：{s['messages']}")
    print(f"完整的state{serializable_state}")
    return serializable_state
        
if __name__ == "__main__":
    print(graph.get_graph(xray=True).draw_mermaid())
