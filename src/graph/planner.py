import json
import logging
import os
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
# from langchain_mcp_adapters.client import MultiServerMCPClient
from src.mcp_client.mcp_client import MultiServerMCPClient_wFileUpload

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan, StepType
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine
from pydantic import BaseModel, Field
import re

logger = logging.getLogger(__name__)

class PlannerOutputSchema(BaseModel):
    """Planner output schema"""
    thought: str = Field(description="思考过程")
    response: str = Field(description="给用户的反馈")
    next_node: str = Field(description="下一步节点的名字")
    instruction: str = Field(description="下一步节点的指令，用自然语言描述")

planner_available_nodes = ["coder", "doc_parser", "web_search", "vqa", "end", "user", "reporter"]

def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["coder", "doc_parser", "web_search", "vqa", "end", "user"]]:
    """Planner node that generate the full plan."""
    logger.info("Inside Planner")
    configurable = Configuration.from_runnable_config(config)
    messages = apply_prompt_template("planner", state, configurable)


    llm = get_llm_by_type(AGENT_LLM_MAP["planner"])
    llm = llm.with_structured_output(PlannerOutputSchema)

    llm_response = llm.invoke(messages)

    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {llm_response}")
    try:
            thought = llm_response.thought
            response = llm_response.response
            next_node = llm_response.next_node
            instruction = llm_response.instruction
    except ValueError as e:
        logger.error(f"Planner parse_result failed: {e}")
        return Command(goto="planner") # retry

    if next_node not in planner_available_nodes:
        logger.error(f"Planner next_node is not in planner_available_nodes: {next_node}")
        return Command(goto="planner") # retry
    else:
        return Command(
            update={
                'current_instruction': instruction,
                'messages': state['messages'] + [AIMessage(content=llm_response.json(), name="planner")],
            },
            goto=next_node,
        )
