"""基础节点抽象类"""

from src.config.agents import AgentConfiguration
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
import logging
from src.config.agents import NodeConfig
from src.graph.tools.tool_manager import ToolManager
from src.prompts.planner_model import Plan
from pprint import pformat

logger = logging.getLogger(__name__)

class BaseNode(ABC):
    """节点基类"""
    
    def __init__(self, name: str, config: 'NodeConfig', tools_manager: 'ToolManager'):
        # self.log_execution("Starting analysis")    
        self.name = name
        self.config = config
        self.tools_manager = tools_manager
        self.iteration_count = 0
        self.call_params = {} # function call 到当前节点需要传入的参数

    @abstractmethod
    async def execute(self, state: Dict[str, Any], config: RunnableConfig) -> Command:
        """执行节点逻辑"""
        pass
       
    def show_current_plan(self, plan: Plan):
        """展示当前计划"""
        logger.info(plan)

    def log_execution(self, message: str):
        """记录执行日志"""
        logger.info(f"[{self.name}] {message}")
    
    def log_input_message(self, message: List):
        """记录输入的信息"""
        logger.info("-" * 50)
        logger.info(f"👇[{self.name}| Input Message]👇")
        for item in message:
            logger.info(f"角色: {item.type}")
            logger.info(f"内容: {item.content}")
            if 'additional_kwargs' in item:
                logger.info(f"附加参数: {item.additional_kwargs}")
            if 'response_metadata' in item:
                logger.info(f"响应元数据: {item.response_metadata}")
            logger.info("-" * 50)

    def log_tool_call(self, response: str, iterate_times: int):
        """记录node的toolcall， 默认逻辑除了planner之外都需要toolcall"""
        logger.info("+" * 50)
        logger.info(f"[{self.name} | iterate time] {iterate_times}")
        logger.info(f"👇[{self.name} | Must Tool Call]👇")
        if 'content' in response:
            logger.info("- 内容:", response.content)
        if 'additional_kwargs' in response:
            logger.info("- 附加参数:", response.additional_kwargs)
        if 'response_metadata' in response:
            logger.info("- 响应元数据:", response.response_metadata)
        if 'tool_calls' in response:
            logger.info("- 工具调用:", response.tool_calls)
        if 'usage_metadata' in response:
            logger.info("- 使用元数据:", response.usage_metadata)
        logger.info("+" * 50)

    def log_execution_warning(self, message: str):
        """记录warnin日志"""
        logger.warning(f"[{self.name}] {message}")

    def log_execution_error(self, message: str):
        """记录error日志"""
        logger.error(f"[{self.name}] {message}")