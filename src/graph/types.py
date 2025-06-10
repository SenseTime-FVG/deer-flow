# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import MessagesState

from src.prompts.planner_model import Plan
from src.rag import Resource


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Runtime Variables
    locale: str = "en-US"
    observations: list[str] = []
    resources: list[Resource] = []
    plan_iterations: int = 0
    current_plan: Plan | str = None
    final_report: str = ""
    auto_accepted_plan: bool = False
    enable_background_investigation: bool = True
    background_investigation_results: str = None
    session_id: str = None
    session_dir: str = None

    # Agent interaction fields
    coder_request: dict = None
    coder_response: str = None

    research_request: dict = None
    research_response: str = None

    reader_request: dict = None
    reader_response: str = None

    analyzer_iteration: int = 0

    delegation_source: str = None
    temp_analysis_result: dict = None