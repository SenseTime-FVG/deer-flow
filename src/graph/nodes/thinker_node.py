# nodes/thinker_node.py
"""思考器节点"""

from .base_node import BaseNode
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from src.config.configuration import Configuration
from src.prompts.template import apply_prompt_template
from typing import Literal, Dict, Any


class ThinkerNode(BaseNode):
    """思考器节点 - 使用推理模型，能够深度思考问题"""
    
    def __init__(self, model, tool_manager):
        super().__init__("thinker", model, tool_manager)
    
    def __call__(self, state: Dict[str, Any], config: RunnableConfig) -> Command[Literal["supervisor"]]:
        """执行思考器逻辑"""
        self.log_execution("Starting deep thinking task")

        configurable = Configuration.from_runnable_config(config)
        messages = apply_prompt_template(
            self.name, state, configurable, state["current_plan"],
            state["current_step_index"])
        response = self.model.invoke(messages, config)
        response.name = self.name

        return Command(
            update={
                "messages": [response],
                "supervisor_iterate_time": state.get("supervisor_iterate_time", 0) + 1
            },
            goto="supervisor"
        )
