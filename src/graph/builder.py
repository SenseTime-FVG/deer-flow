# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .types import State
from .nodes import (
    reporter_node,
    background_investigation_node,
    end_node
)
from .planner import planner_node
from .user import user_node
from .coder import coder_node
from .web_search import web_search_node
from .doc_parser import doc_parser_node
from .vqa import vqa_node


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "planner")
    builder.add_node("planner", planner_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("doc_parser", doc_parser_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("vqa", vqa_node)
    builder.add_node("coder", coder_node)
    builder.add_node("user", user_node)
    builder.add_node("end", end_node)
    builder.add_edge("end", END)
    return builder


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    memory = MemorySaver()
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


graph = build_graph()
