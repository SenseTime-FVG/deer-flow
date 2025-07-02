import json
import logging
import os
import time
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
from jinja2 import Environment, FileSystemLoader, select_autoescape
import re
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class UserOutputSchema(BaseModel):
    thought: str
    question: str


def get_prompt(instruction: str, history: list):
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)+"/../prompts"),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("user.md")
    return template.render(instruction=instruction, history=history)


def user_node(
    state, config
) -> Command[Literal["planner"]]:
    logger.info("Inside Planner")
    configurable = Configuration.from_runnable_config(config)

    instruction = state["current_instruction"]
    history = state["messages"]
    prompt = get_prompt(instruction, history)

    llm = get_llm_by_type(AGENT_LLM_MAP["user"])
    llm = llm.with_structured_output(UserOutputSchema)
    llm_response = llm.invoke(prompt)
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"response: {llm_response}")
    
    # parse the result
    try:
        thought = llm_response.thought
        question = llm_response.question
    except ValueError as e:
        logger.error(f"user_node parse_result failed: {e}")
        time.sleep(2) # todo: 重试
        return Command(goto="user") # retry

    # 获取用户反馈
    feedback = interrupt(question)
    logger.info(f'received feedback: {feedback}')


    return Command(
        goto="planner",
        update={
            'messages': [
                AIMessage(content=llm_response.json(), name="user"),
                HumanMessage(content=feedback, name="feedback")
            ]
        }
    )
