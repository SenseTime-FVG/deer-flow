import json
import logging
import os
import os.path as osp
import uuid
import filetype
import traceback
from copy import deepcopy
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.types import Command, interrupt
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId


from langchain_core.runnables import RunnableConfig
# from langchain_mcp_adapters.client import MultiServerMCPClient
from src.mcp_client.mcp_client import MultiServerMCPClient_wFileUpload

from src.agents import create_agent
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
)

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan, StepType
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine

from .nodes import _setup_and_execute_agent_step
import requests

logger = logging.getLogger(__name__)

class CustomAgentState(AgentState):
    """The state of the agent with a session directory."""

    session_dir: str
    resources: list[dict]

async def web_search_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner"]]:
    """web_search_node"""
    logger.info("inside web_search_node")
    configurable = Configuration.from_runnable_config(config)
    # tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
    tools = [get_web_search_tool(configurable.max_search_results), download_file]
    logger.info(f"web_search tools: {tools}")
    return await _setup_and_execute_agent_step(
        state,
        config,
        "web_search",
        tools,
    )

# class DownloadToolArgsSchema(BaseModel):
#     """Download tool args schema"""
#     url: str | list[str] = Field(description="The url or url list to download")
#     state: Annotated[AgentStateWithSessionDir, InjectedState] = Field(description="The state of the agent",)

@tool("download_file")
def download_file(
    urls: str | list[str],
    state: Annotated[CustomAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """
    你可以用这个工具下载网络上的文件
    Args:
        urls: The url or url list to download
        state: The state of the agent
    
    Returns:
        A message indicating the success or failure of the download
    """
    logger.info(f"download_file is called with urls: {urls}")
    try:
        return _download_file(urls, state, tool_call_id)
    except Exception as e:
        logger.error(traceback.format_exc())
        return f"Error while downloading file with urls: {urls}: {e}"

def _download_file(urls, state, tool_call_id):
    session_dir = state['session_dir']
    if isinstance(urls, str):
        urls = [urls]
    
    success_message = ""
    fail_message = ""
    new_resources = []
    for url in urls:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            output_file_path = osp.join(session_dir, f"{uuid.uuid4().hex[:8]}.{filetype.guess(response.content).extension}")
            with open(output_file_path, "wb") as f:
                f.write(response.content)

            new_resources.append(dict(
                uri=output_file_path,
                title=output_file_path,
                description=output_file_path,
                resource_id=len(state['resources'])
            ))
            success_message += f"File downloaded successfully: {url} -> {output_file_path}\n"
        except Exception as e:
            logger.error(f"Error downloading file {url}: {e}")
            fail_message += f"Error downloading file {url}: {e}\n"


    return Command(
        update={
            "messages": state['messages'] + [
                ToolMessage(
                    content=f"{success_message}\n\n{fail_message}",
                    tool_call_id=tool_call_id
                )
            ],
            "resources": state['resources'] + new_resources
        }
    )
    
    
def pre_model_hook(state, config):
    if 'remaining_steps' in state and state['remaining_steps'] <= 3:
        return Command(update={
            "messages": state['messages'] + [
                HumanMessage(content="因为所剩步数不足，你现在不能继续执行工具，现在应当开始进行总结。\n请根据当前的搜索结果，输出你的结论。")
            ]}
        )
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
    agent = create_agent(agent_type, agent_type, loaded_tools, agent_type, pre_model_hook, CustomAgentState)
    return await _execute_agent_step(state, agent, agent_type)


async def _execute_agent_step(
    state: State, agent, agent_name: str
) -> Command[Literal["planner"]]:

    instruction = state['current_instruction']
    # Prepare the input for the agent with completed steps info
    agent_input = {
        "session_dir": state['session_dir'],
        "resources": [],
        "messages": [
            HumanMessage(
                content=instruction
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
