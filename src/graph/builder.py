# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import StateGraph, START, END
from src.graph.tools.tool_manager import ToolManager
from src.graph.types import State
from src.graph.nodes import (
    CoordinatorNode,
    PlannerNode,
    WriterNode,
    CoderNode,  
    InterpreterNode,
    SearcherNode,
    ReaderNode,
    SupervisorNode,  
    ReporterNode,  
    ReceiverNode
)

import logging

logger = logging.getLogger(__name__)

def _build_base_graph(
    starter: str = "coordinator",
    model_dict: dict = {}, tools: list = [], node_tools: dict = {}
):
    tool_manager = ToolManager(tools, node_tools)
    nodes = {
        "coordinator": CoordinatorNode(
            model_dict["basic"], tool_manager),
        "planner": PlannerNode(
            model_dict["basic"], tool_manager),
        "writer": WriterNode(model_dict["basic"], tool_manager),
        "coder": CoderNode(model_dict["basic"], tool_manager),
        "interpreter": InterpreterNode(
            model_dict["basic"], tool_manager),
        "searcher": SearcherNode(model_dict["basic"], tool_manager),
        "reader": ReaderNode(model_dict["basic"], tool_manager),
        "receiver": ReceiverNode(model_dict["basic"], tool_manager),
        "supervisor": SupervisorNode(
            model_dict["basic"], tool_manager),
        "reporter": ReporterNode(model_dict["basic"], tool_manager),
    }

    """Build the agent workflow graph with all nodes."""
    builder = StateGraph(State)

    # add model nodes
    for node_name, node_instance in nodes.items():
        builder.add_node(node_name, node_instance)

    # add tool nodes and returning edges
    builder = tool_manager.build_tool_nodes_and_edges(builder, nodes)
    builder.add_edge(START, starter)\
        .add_edge("supervisor", END)

    return builder


def build_graph_from_config(build_args: dict = {}, compile_args: dict = {}):
    builder = _build_base_graph(**build_args)
    return builder.compile(**compile_args)

# graph = build_graph_with_memory()
