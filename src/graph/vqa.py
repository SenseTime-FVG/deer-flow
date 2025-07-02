import json
import logging
import os
import os.path as osp
import base64
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
# from langchain_mcp_adapters.client import MultiServerMCPClient
from src.mcp_client.mcp_client import MultiServerMCPClient_wFileUpload
from src.agents import create_agent

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan, StepType
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .nodes import _setup_and_execute_agent_step
from src.utils.file_utils import file_to_data_uri

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field
from copy import deepcopy
# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)

class VQAInputSchema(BaseModel):
    """Ask a question about the images"""
    question: str = Field(description="Question to ask about the images")
    images: list[str] = Field(description="The image paths to ask the question about")

@tool("vqa_model", description="Ask a question about the images", args_schema=VQAInputSchema)
def vqa_model(question, images) -> str:
        
    query_content = [{"type": "text", "text": question}]
    for image in images:
        if not osp.exists(image):
            logger.warning(f"Image {image} does not exist")
            full_response = f'One of the image does not exist: {image}'
            return full_response
        else:
            query_content.append({"type": "image_url", "image_url": {"url": file_to_data_uri(image)}})
    
    messages = [HumanMessage(content=query_content)]

    llm = get_llm_by_type(AGENT_LLM_MAP["vqa"])
    
    if AGENT_LLM_MAP["user"] == "basic":
        full_response = llm.invoke(messages)
    else:
        response = llm.stream(messages)
        full_response = ""
        for chunk in response:
            full_response += chunk.content
    return full_response



def get_prompt(instruction: str, history: list):
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("user.md")
    return template.render(instruction=instruction, history=history)


async def vqa_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner"]]:
    logger.info("inside vqa_node")
    configurable = Configuration.from_runnable_config(config)
    tools = [vqa_model]
    logger.info(f"vqa_node tools: {tools}")
    return await _setup_and_execute_agent_step(
        state,
        config,
        "vqa",
        tools,
    )


def pre_model_hook(state, config):
    if 'remaining_steps' in state and state['remaining_steps'] <= 3:
        return {
            "messages": [
                HumanMessage(content="因为所剩步数不足，你现在不能继续执行工具，现在应当开始进行总结。\n请根据当前的结果，输出你的结论。")
            ]
        }
    return state

async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_type: str,
    default_tools: list,
) -> Command[Literal["planner"]]:
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = {}
    enabled_tools = {}

    # Extract MCP server configuration for this agent type
    if configurable.mcp_settings:
        for server_name, server_config in configurable.mcp_settings["servers"].items():
            if (
                server_config["enabled_tools"]
                and agent_type in server_config["add_to_agents"]
            ):
                mcp_servers[server_name] = {
                    k: v
                    for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env")
                }
                for tool_name in server_config["enabled_tools"]:
                    enabled_tools[tool_name] = server_name

    # Create and execute agent with MCP tools if available
    loaded_tools = deepcopy(default_tools)
    if mcp_servers:
        async with MultiServerMCPClient_wFileUpload(mcp_servers) as client:
            for tool in client.get_tools():
                if tool.name in enabled_tools:
                    tool.description = (
                        f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
                    )
                    loaded_tools.append(tool)
    logger.info(f'agent_type={agent_type}, loaded_tools={loaded_tools}')
    agent = create_agent(agent_type, agent_type, loaded_tools, agent_type, pre_model_hook)
    return await _execute_agent_step(state, agent, agent_type)


async def _execute_agent_step(
    state: State, agent, agent_name: str
) -> Command[Literal["planner"]]:

    instruction = state['current_instruction']
    # Prepare the input for the agent with completed steps info
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"来自Planner的指令/问题： {instruction}"
            )
        ],
    }

    # Invoke the agent
    default_recursion_limit = 5
    recursion_limit = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
    recursion_limit = int(recursion_limit)
    assert recursion_limit > 0, f"Recursion limit must be positive, but got {recursion_limit}"
    logger.info(f"Recursion limit set to: {recursion_limit}")

    logger.info(f"Agent input: {agent_input}")
    result = await agent.ainvoke(
        input=agent_input, config={"recursion_limit": recursion_limit}
    )

    # Process the result
    response_content = result["messages"][-1].content
    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")


    return Command(
        update={
            "messages": state['messages'] + [
                AIMessage(
                    content=response_content,
                    name=agent_name,
                )
            ],
            "observations": state.get('observations', []) + [response_content]
        },
        goto="planner",
    )
