# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.graph.tools.tool_manager import ToolManager
from .types import State
from .nodes import (
    CoordinatorNode,
    PlannerNode,
    WriterNode,
    CoderNode,  
    InterpreterNode,
    SearcherNode,
    ReaderNode,
    ThinkerNode,
    SupervisorNode,  
    ReporterNode,  
    ReceiverNode
)
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)

def _build_base_graph():
    
    # 全局工具管理器实例
    tool_manager = ToolManager()    
    nodes = {
        "coordinator": CoordinatorNode(tool_manager),
        "planner": PlannerNode(tool_manager),
        "writer": WriterNode(tool_manager),
        "coder": CoderNode(tool_manager),
        "interpreter": InterpreterNode(tool_manager),
        "searcher": SearcherNode(tool_manager),
        "reader": ReaderNode(tool_manager),
        # "thinker": ThinkerNode(tool_manager),
        "receiver": ReceiverNode(tool_manager),
        "supervisor": SupervisorNode(tool_manager),
        "reporter": ReporterNode(tool_manager),
    }
    logger.info(f"Initialized {len(nodes)} nodes")

    """Build the agent workflow graph with all nodes."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")

    for node_name, node_instance in nodes.items():
        builder.add_node(node_name, node_instance.execute)
        tool_manager.register_tool(f"call_{node_name}_agent", node_instance.call_params)
        logger.debug(f"Added node: {node_name}")

    builder.add_edge("supervisor", END)
    # 输出统计信息
    stats = tool_manager.get_statistics()
    logger.info(f"Tool initialization complete. Stats: {stats}")

    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


def build_graph_from_config(compile_args):
    builder = _build_base_graph()
    return builder.compile(**compile_args)


def build_searcher_subgraph_with_memory():
    """
    构建并返回一个带有记忆功能的 Searcher 节点子图。
    该子图在达到 'supervisor' 状态时结束。
    """
    # 实例化 MemorySaver，用于在内存中保存图的状态
    memory_builder = MemorySaver()

    # 构建状态图
    builder_searcher = StateGraph(State)

    tool_manager = ToolManager()
    searcher_node = SearcherNode(tool_manager)
    # supervisor_node = SupervisorNode(tool_manager)

    builder_searcher.add_node("searcher", searcher_node.execute)
    # builder_searcher.add_node("supervisor", supervisor_node.execute)
    # 设置入口点
    builder_searcher.set_entry_point("searcher")

    # 定义条件边，根据 SearcherNode 的 Command.goto 决定流向
    builder_searcher.add_conditional_edges(
        "searcher",
        lambda x: decide_next_node(x), # 这个lambda函数直接返回节点执行后Command.goto的值
        {
            "searcher": "searcher", # 如果返回 "searcher"，则循环回 "searcher" 节点
            "__end__": END,         # 如果返回 "__end__"，则子图结束
        },
    )

    # 编译图，并将 MemorySaver 实例作为 checkpointer 传入
    return builder_searcher.compile(checkpointer=memory_builder)

def decide_next_node(x):
    # print(f"返回值：{x}")  # 保留你的打印方便调试

    # 举例：如果 tool message 里有“search_results”，就继续搜
    if isinstance(x, dict):
        messages = x.get("messages", [])
        last_message = messages[-1]
        if "search_results" in last_message.content:
            return "searcher"
        else:
            return "__end__"