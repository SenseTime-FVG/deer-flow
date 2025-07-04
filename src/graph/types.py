# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages

from src.prompts.planner_model import Plan
from src.rag import Resource
from typing import Annotated, Dict


def merge_dict(left: dict, right: dict):
    intersect_keys = set(left.keys()) & set(right.keys())
    for i in intersect_keys:
        if isinstance(left[i], dict) and isinstance(right[i], dict):
            left[i] = merge_dict(left[i], right[i])

    for i in right:
        if i not in left:
            left[i] = right[i]

    return left


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""
    # 如果不在此定义即使后面有定义也不会更新，不会报错但是不会有新加的变量
    # Runtime Variables

    # model context
    task_messages: Annotated[list[AnyMessage], add_messages]
    plan_messages: Annotated[list[AnyMessage], add_messages]
    action_message: Annotated[list[AnyMessage], add_messages]
    supervisor_message: Annotated[list[AnyMessage], add_messages]

    # finalized message (including history prompt)
    default_task_messages: list[AnyMessage]
    default_plan_messages: list[AnyMessage]
    default_action_messages: Annotated[
        Dict[str, Dict[str, list[AnyMessage]]], merge_dict
    ]

    locale: str = "zh-CN"
    observations: list[str] = []
    resources: list[Resource] = []
    plan_iterations: int = 0
    current_plan: Plan | str = None
    current_step_index: str = None
    final_report: str = ""
    auto_accepted_plan: bool = False

    enable_background_investigation: bool = True
    background_investigation_results: str = None
    session_id: str = None
    session_dir: str = None

    tool_call_iterate_time: int = 0  # 当前toolcall的迭代次数
    supervisor_iterate_time: int = 0  # supervisor_iterate_time
    history_clear: bool = False  # 是否清空

    file_info: str
    need_image: str = "true"
