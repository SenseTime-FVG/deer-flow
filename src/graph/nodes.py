# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.mcp_client.mcp_client import MultiServerMCPClient_wFileUpload
from src.agents import create_agent
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
)

from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan, StepType
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from src.graph.types import State
from src.config import SELECTED_SEARCH_ENGINE, SearchEngine

logger = logging.getLogger(__name__)

# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision"]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "coordinator": "basic",
    "planner": "basic", 
    "router": "basic",
    "analyzer": "basic",
    "coder": "basic",
    "researcher": "basic",
    "reader": "vision",
    "thinker": "reasoning",
    "reporter": "basic",
}

@tool
def handoff_to_planner(
    task_title: Annotated[str, "The title of the task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    return

@tool
def call_coder_agent(
    analysis_request: Annotated[str, "Specific analysis request for the coder"],
    file_info: Annotated[str, "Information about files to be processed"] = "",
):
    """Request coder agent to handle coding and data analysis tasks."""
    return

@tool
def call_researcher_agent(
    research_request: Annotated[str, "Specific research request for the researcher"],
    search_scope: Annotated[str, "Scope and focus of the research"] = "",
):
    """Request researcher agent to handle information gathering and research tasks."""
    return

@tool
def call_reader_agent(
    reader_request: Annotated[str, "Specific request for the reader to process documents/images"],
    file_info: Annotated[str, "Information about files to be processed"] = "",
):
    """Request reader agent to handle document and image processing tasks."""
    return

def _summarize_step_execution(step_title: str, agent_messages: list, llm_type: str = "basic") -> str:
    """Summarize step execution when agent interactions exceed 5 calls."""
    logger.info(f"Summarizing step execution for: {step_title}")
    
    # Extract content from agent messages (exclude system prompts)
    content_messages = []
    for msg in agent_messages:
        if hasattr(msg, 'content') and msg.content:
            # Skip system prompts and very short messages
            if len(msg.content) > 50 and not msg.content.startswith("You are"):
                content_messages.append(msg.content)
    
    if not content_messages:
        return f"Step '{step_title}' completed with no significant content."
    
    # Create summarization prompt
    summary_content = "\n\n".join(content_messages)
    summary_prompt = f"""
Please provide a concise summary of the following step execution:

## Step: {step_title}

## Execution Content:
{summary_content}

## Summary Requirements:
- Focus on key findings and results
- Include important data points and insights
- Mention any decisions or conclusions reached
- Keep it comprehensive but concise
- Maintain factual accuracy

Please provide a well-structured summary that captures the essential outcomes of this step.
"""
    
    try:
        llm = get_llm_by_type(llm_type)
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        return f"## Step Summary: {step_title}\n\n{summary_response.content}"
    except Exception as e:
        logger.error(f"Error summarizing step {step_title}: {e}")
        # Fallback to simple concatenation
        return f"## Step: {step_title}\n\n" + "\n\n".join(content_messages[:3])  # Take first 3 messages

def _prepare_step_memory(state: State, current_step) -> str:
    """Prepare memory context from previous steps for current step execution."""
    current_plan = state.get("current_plan")
    if not current_plan:
        return ""
    
    step_memories = state.get("step_memories", {})
    memory_context = ""
    
    if step_memories:
        memory_context = "# Previous Step Results\n\n"
        for step_id, memory in step_memories.items():
            memory_context += f"{memory}\n\n"
    
    # Add current delegation responses if available
    # coder_response = state.get("coder_response")
    # research_response = state.get("research_response")
    # reader_response = state.get("reader_response")
    
    # if coder_response:
    #     memory_context += f"# Recent Coder Analysis\n\n{coder_response}\n\n"
    
    # if research_response:
    #     memory_context += f"# Recent Research Results\n\n{research_response}\n\n"
        
    # if reader_response:
    #     memory_context += f"# Recent Reader Analysis\n\n{reader_response}\n\n"
    
    return memory_context

def _update_step_memory(state: State, step_title: str, agent_messages: list, interaction_count: int):
    """Update step memory based on interaction count."""
    step_memories = state.get("step_memories", {})
    step_id = f"step_{len(step_memories) + 1}"
    
    if interaction_count > 5:
        # Summarize if too many interactions
        step_memory = _summarize_step_execution(step_title, agent_messages)
    else:
        # Direct concatenation for fewer interactions
        content_messages = []
        for msg in agent_messages:
            if hasattr(msg, 'content') and msg.content and len(msg.content) > 50:
                # Skip system prompts
                if not msg.content.startswith("You are") and not msg.content.startswith("---"):
                    content_messages.append(msg.content)
        
        if content_messages:
            step_memory = f"## Step: {step_title}\n\n" + "\n\n".join(content_messages)
        else:
            step_memory = f"## Step: {step_title}\n\nCompleted with no significant output."
    
    step_memories[step_id] = step_memory
    return step_memories

def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "__end__"]]:
    """协调器节点 - 过滤无意义问题"""
    logger.info("Coordinator node is running.")
    configurable = Configuration.from_runnable_config(config)
    messages = apply_prompt_template("coordinator", state, configurable)
    
    response = (
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools([handoff_to_planner])
        .invoke(messages)
    )
    
    logger.debug(f"Coordinator response: {response}")
    
    goto = "__end__"
    locale = state.get("locale", "en-US")
    
    if len(response.tool_calls) > 0:
        goto = "planner"
        try:
            for tool_call in response.tool_calls:
                if tool_call.get("name", "") != "handoff_to_planner":
                    continue
                if tool_locale := tool_call.get("args", {}).get("locale"):
                    locale = tool_locale
                    break
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
    else:
        logger.warning("Coordinator response contains no tool calls. Terminating workflow.")
    
    return Command(
        update={
            "locale": locale, 
            "resources": state["resources"],
            "messages": [AIMessage(content=response.content, name="coordinator")],
            "step_memories": {}  # Initialize step memories
        },
        goto=goto,
    )

def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["router", "__end__"]]: #, "reporter"
    """规划器节点 - 计划 → router"""
    logger.info("Planner node is generating execution plan.")
    configurable = Configuration.from_runnable_config(config)
    
    messages = apply_prompt_template("planner", state, configurable)
    
    # 背景调研
    if state.get("enable_background_investigation", True):
        query = state["messages"][-1].content
        if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
            searched_content = LoggedTavilySearch(
                max_results=configurable.max_search_results
            ).invoke(query)
            if isinstance(searched_content, list):
                background_results = [
                    {"title": elem["title"], "content": elem["content"]}
                    for elem in searched_content
                ]
                messages.append({
                    "role": "user",
                    "content": f"Background investigation results:\n{json.dumps(background_results, ensure_ascii=False)}"
                })
    
    llm = get_llm_by_type(AGENT_LLM_MAP["planner"]).with_structured_output(
        Plan, method="json_mode"
    )
    
    response = llm.invoke(messages)
    plan_content = response.model_dump_json(indent=4, exclude_none=True)
    
    logger.info(f"Generated plan: {plan_content}")
    
    try:
        plan_dict = json.loads(plan_content)
        new_plan = Plan.model_validate(plan_dict)
        # 不直接走reporter了，无论怎么样都给到analyzer
        # # 检查是否有足够的上下文直接生成报告 TODO
        # if plan_dict.get("has_enough_context", False):
        #     return Command(
        #         update={
        #             "current_plan": new_plan,
        #             "messages": [AIMessage(content=plan_content, name="planner")]
        #         },
        #         goto="reporter"
        #     )
        # else:
        return Command(
            update={
                "current_plan": new_plan,
                "messages": [AIMessage(content=plan_content, name="planner")]
            },
            goto="router"
            )
    except json.JSONDecodeError:
        logger.error("Failed to parse plan JSON")
        return Command(goto="__end__")

def router_node(
    state: State
) -> Command[Literal["analyzer", "coder", "researcher", "reader", "thinker", "planner", "reporter"]]:
    """路由节点 - 根据planner执行step（按逻辑中转，无LLM）"""
    logger.info("Router node is dispatching tasks.")
    
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        logger.warning("No plan or steps available, routing to planner")
        return Command(goto="planner")
    
    # 检查是否所有步骤都已完成
    if all(step.execution_res for step in current_plan.steps):
        logger.info("All steps completed, routing to reporter")
        return Command(goto="reporter")
    
    # 找到第一个未完成的步骤
    current_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            current_step = step
            break
    
    if not current_step:
        return Command(goto="reporter")
    
    # 根据步骤类型路由到相应的agent
    step_type = current_step.step_type.lower()
    
    if step_type == "analyzer":
        logger.info(f"Routing to analyzer for step: {current_step.title}")
        return Command(goto="analyzer")
    elif step_type == "coder":
        logger.info(f"Routing to coder for step: {current_step.title}")
        return Command(goto="coder")
    elif step_type == "researcher":
        logger.info(f"Routing to researcher for step: {current_step.title}")
        return Command(goto="researcher")
    elif step_type == "reader":
        logger.info(f"Routing to reader for step: {current_step.title}")
        return Command(goto="reader")
    elif step_type == "thinker":
        logger.info(f"Routing to thinker for step: {current_step.title}")
        return Command(goto="thinker")
    else:
        # 默认路由到analyzer
        logger.info(f"Default routing to analyzer for step: {current_step.title}")
        return Command(goto="analyzer")

async def analyzer_node(
    state: State, config: RunnableConfig
) -> Command[Literal["router", "coder", "researcher", "reader"]]:
    
    """分析器节点 - 代码和数据任务→Coder Agent, 研究和搜索任务→Researcher Agent, 图像理解→Reader Agent"""
    logger.info("Analyzer node is coordinating task execution.")
    configurable = Configuration.from_runnable_config(config)
    
    current_plan = state.get("current_plan")
    current_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            current_step = step
            break
    
    if not current_step:
        return Command(goto="router")
    
    # 准备带有步骤记忆的输入
    memory_context = _prepare_step_memory(state, current_step)
    print(state["resources"])
    analyzer_input = {
        "messages": [
            HumanMessage(
                content=f"{memory_context}# Current Task\n\n## Title\n{current_step.title}\n\n## Description\n{current_step.description}\n\n## Locale\n{state.get('locale', 'en-US')}"
            )
        ],
        "locale": state.get("locale", "en-US"),
        "resources": state.get("resources", [])
    }
    
    
    messages = apply_prompt_template("analyzer", analyzer_input, configurable)
    print(f"analyzer: \n{messages}")
    # 准备基础工具
    tools = [
        call_coder_agent,
        call_researcher_agent,
        call_reader_agent
    ]
    
    # 处理MCP服务器配置，为analyzer添加MCP工具
    mcp_servers = {}
    if configurable.mcp_settings:
        for server_name, server_config in configurable.mcp_settings["servers"].items():
            if (
                server_config.get("enabled_tools")
                and "analyzer" in server_config.get("add_to_agents", [])
            ):
                mcp_servers[server_name] = {
                    k: v for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env")
                }
    
    # 创建agent并执行
    if mcp_servers:
        async with MultiServerMCPClient_wFileUpload(mcp_servers, state=state) as client:
            loaded_tools = tools[:]
            for tool in client.get_tools():
                loaded_tools.append(tool)
            agent = create_agent("analyzer", "analyzer", loaded_tools, "analyzer")
            
            recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", "10"))
            result = await agent.ainvoke(
                input={"messages": messages},
                config={"recursion_limit": recursion_limit}
            )
        
        response = result["messages"][-1]
        agent_messages = result.get("messages", [])
    else:
        # 只使用基础工具
        llm = get_llm_by_type(AGENT_LLM_MAP["analyzer"]).bind_tools(tools)
        response = llm.invoke(messages)
        agent_messages = [response]
    
    print(f"analyzer response: {response}")
    
    # 检查是否需要委托给其他agent
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "call_coder_agent":
                coder_request = {
                    "analysis_request": tool_call["args"]["analysis_request"],
                    "file_info": tool_call["args"].get("file_info", ""),
                    "step_context": current_step,
                    "analyzer_response": response.content
                }
                return Command(
                    update={
                        "coder_request": coder_request,
                        "delegation_source": "analyzer",
                        "current_agent_messages": agent_messages
                    },
                    goto="coder"
                )
            elif tool_call["name"] == "call_researcher_agent":
                research_request = {
                    "research_request": tool_call["args"]["research_request"],
                    "search_scope": tool_call["args"].get("search_scope", ""),
                    "step_context": current_step,
                    "analyzer_response": response.content
                }
                return Command(
                    update={
                        "research_request": research_request,
                        "delegation_source": "analyzer",
                        "current_agent_messages": agent_messages
                    },
                    goto="researcher"
                )
            elif tool_call["name"] == "call_reader_agent":
                reader_request = {
                    "reader_request": tool_call["args"]["reader_request"],
                    "file_info": tool_call["args"].get("file_info", ""),
                    "step_context": current_step,
                    "analyzer_response": response.content
                }
                return Command(
                    update={
                        "reader_request": reader_request,
                        "delegation_source": "analyzer",
                        "current_agent_messages": agent_messages
                    },
                    goto="reader"
                )
    
    # 没有委托，直接完成步骤
    current_step.execution_res = response.content
    
    # 更新步骤记忆
    step_memories = _update_step_memory(state, current_step.title, agent_messages, len(agent_messages))
    
    return Command(
        update={
            "messages": [AIMessage(content=response.content, name="analyzer")],
            "step_memories": step_memories,
            "observations": state.get("observations", []) + [response.content]
        },
        goto="router"
    )

async def coder_node(
    state: State, config: RunnableConfig
) -> Command[Literal["router", "analyzer"]]:
    """编程节点 - 处理analyzer请求→Analyzer, 根据任务生成和执行代码<->self"""
    logger.info("Coder node is executing coding task.")
    
    # 检查是否来自analyzer的委托
    coder_request = state.get("coder_request")
    if coder_request:
        logger.info("Processing analyzer's coder request")
        return await _execute_coder_for_analyzer(state, config)
    else:
        # 直接执行编程任务
        return await _execute_coder_step(state, config)

async def _execute_coder_step(
    state: State, config: RunnableConfig
) -> Command[Literal["router"]]:
    """执行常规coder步骤"""
    logger.info("Executing regular coder step")
    configurable = Configuration.from_runnable_config(config)
    
    current_plan = state.get("current_plan")
    current_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            current_step = step
            break
    
    if not current_step:
        return Command(goto="router")
    
    # 准备带有步骤记忆的输入
    memory_context = _prepare_step_memory(state, current_step)
    coder_input = {
        "messages": [
            HumanMessage(
                content=f"{memory_context}# Current Task\n\n## Title\n{current_step.title}\n\n## Description\n{current_step.description}\n\n## Locale\n{state.get('locale', 'en-US')}"
            )
        ],
        "locale": state.get("locale", "en-US"),
        "resources": state.get("resources", [])
    }
    
    messages = apply_prompt_template("coder", coder_input, configurable)
    
    # 创建coder agent
    coder_agent = create_agent("coder", "coder", [python_repl_tool], "coder")
    
    # 执行agent并跟踪交互次数
    recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", "30"))
    result = await coder_agent.ainvoke(
        input={"messages": messages},
        config={"recursion_limit": recursion_limit}
    )
    
    execution_result = result["messages"][-1].content
    current_step.execution_res = execution_result
    
    # 估算交互次数（基于消息数量）
    interaction_count = len(result.get("messages", []))
    
    # 更新步骤记忆
    step_memories = _update_step_memory(state, current_step.title, result.get("messages", []), interaction_count)
    
    return Command(
        update={
            "messages": [AIMessage(content=execution_result, name="coder")],
            "step_memories": step_memories,
            "observations": state.get("observations", []) + [execution_result]
        },
        goto="router"
    )

async def _execute_coder_for_analyzer(
    state: State, config: RunnableConfig
) -> Command[Literal["analyzer"]]:
    """为analyzer执行代码分析任务"""
    logger.info("Executing coder task for analyzer")
    configurable = Configuration.from_runnable_config(config)
    
    coder_request = state.get("coder_request")
    
    # 构建coder任务
    memory_context = _prepare_step_memory(state, coder_request['step_context'])
    coder_task = f"""
{memory_context}

# Task from Analyzer

## Analysis Request
{coder_request['analysis_request']}

## File Information
{coder_request['file_info']}

## Context from Analyzer
{coder_request['analyzer_response']}

## Step Context
{coder_request['step_context'].description}

Please complete this coding/data analysis task and return comprehensive results.
"""
    
    coder_input = {
        "messages": [HumanMessage(content=coder_task)],
        "locale": state.get("locale", "en-US")
    }
    
    messages = apply_prompt_template("coder", coder_input, configurable)
    
    # 创建并执行coder agent
    coder_agent = create_agent("coder", "coder", [python_repl_tool], "coder")
    recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", "30"))
    
    result = await coder_agent.ainvoke(
        input={"messages": messages},
        config={"recursion_limit": recursion_limit}
    )
    
    coder_response = result["messages"][-1].content
    
    return Command(
        update={
            "coder_response": coder_response,
            "messages": [AIMessage(content=coder_response, name="coder")],
            "coder_request": None  # 清除请求
        },
        goto="analyzer"
    )

async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["router", "analyzer"]]:
    """研究员节点 - 资料查找和分析<->self"""
    logger.info("Researcher node is executing research task.")
    configurable = Configuration.from_runnable_config(config)
    
    # 检查是否来自analyzer的委托
    research_request = state.get("research_request")
    if research_request:
        logger.info("Processing analyzer's research request")
        return await _execute_researcher_for_analyzer(state, config)
    else:
        # 直接执行研究任务
        tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
        retriever_tool = get_retriever_tool(state.get("resources", []))
        if retriever_tool:
            tools.insert(0, retriever_tool)
        
        return await _execute_agent_step(state, config, "researcher", tools)

async def _execute_researcher_for_analyzer(
    state: State, config: RunnableConfig
) -> Command[Literal["analyzer"]]:
    """为analyzer执行研究任务"""
    logger.info("Executing research task for analyzer")
    configurable = Configuration.from_runnable_config(config)
    
    research_request = state.get("research_request")
    
    # 构建研究任务
    memory_context = _prepare_step_memory(state, research_request['step_context'])
    research_task = f"""
{memory_context}

# Research Task from Analyzer

## Research Request
{research_request['research_request']}

## Search Scope
{research_request['search_scope']}

## Context from Analyzer
{research_request['analyzer_response']}

## Step Context
{research_request['step_context'].description}

Please complete this research task and return comprehensive findings.
"""
    
    research_input = {
        "messages": [HumanMessage(content=research_task)],
        "locale": state.get("locale", "en-US"),
        "resources": state.get("resources", [])
    }
    
    messages = apply_prompt_template("researcher", research_input, configurable)
    
    # 准备研究工具
    tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
    retriever_tool = get_retriever_tool(state.get("resources", []))
    if retriever_tool:
        tools.insert(0, retriever_tool)
    
    # 创建并执行researcher agent
    researcher_agent = create_agent("researcher", "researcher", tools, "researcher")
    recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", "25"))
    
    result = await researcher_agent.ainvoke(
        input={"messages": messages},
        config={"recursion_limit": recursion_limit}
    )
    
    research_response = result["messages"][-1].content
    
    return Command(
        update={
            "research_response": research_response,
            "messages": [AIMessage(content=research_response, name="researcher")],
            "research_request": None  # 清除请求
        },
        goto="analyzer"
    )

async def reader_node(
    state: State, config: RunnableConfig
) -> Command[Literal["router", "analyzer"]]:
    """读取器节点 - 读文件，返回caption→Analyzer"""
    logger.info("Reader node is processing document content.")
    configurable = Configuration.from_runnable_config(config)
    
    # 检查是否来自analyzer的委托
    reader_request = state.get("reader_request")
    if reader_request:
        logger.info("Processing analyzer's reader request")
        return await _execute_reader_for_analyzer(state, config)
    else:
        # 直接执行读取任务
        current_plan = state.get("current_plan")
        current_step = None
        for step in current_plan.steps:
            if not step.execution_res:
                current_step = step
                break
        
        if not current_step:
            return Command(goto="router")
        
        # 准备带有步骤记忆的输入
        memory_context = _prepare_step_memory(state, current_step)
        reader_input = {
            "messages": [
                HumanMessage(
                    content=f"{memory_context}# Current Task\n\n## Title\n{current_step.title}\n\n## Description\n{current_step.description}\n\n## Locale\n{state.get('locale', 'en-US')}"
                )
            ],
            "locale": state.get("locale", "en-US"),
            "resources": state.get("resources", [])
        }
        
        messages = apply_prompt_template("reader", reader_input, configurable)
        
        # 使用vision模型处理文档和图片
        llm = get_llm_by_type(AGENT_LLM_MAP["reader"])
        response = llm.invoke(messages)
        
        current_step.execution_res = response.content
        
        # 更新步骤记忆
        step_memories = _update_step_memory(state, current_step.title, [response], 1)
        
        return Command(
            update={
                "messages": [AIMessage(content=response.content, name="reader")],
                "step_memories": step_memories,
                "observations": state.get("observations", []) + [response.content]
            },
            goto="router"
        )

async def _execute_reader_for_analyzer(
    state: State, config: RunnableConfig
) -> Command[Literal["analyzer"]]:
    """为analyzer执行文档读取任务"""
    logger.info("Executing reader task for analyzer")
    configurable = Configuration.from_runnable_config(config)
    
    reader_request = state.get("reader_request")
    
    # 构建读取任务
    memory_context = _prepare_step_memory(state, reader_request['step_context'])
    reader_task = f"""
{memory_context}

# Document Processing Task from Analyzer

## Reader Request
{reader_request['reader_request']}

## File Information
{reader_request['file_info']}

## Context from Analyzer
{reader_request['analyzer_response']}

## Step Context
{reader_request['step_context'].description}

Please process the documents/images and return comprehensive analysis.
"""
    
    reader_input = {
        "messages": [HumanMessage(content=reader_task)],
        "locale": state.get("locale", "en-US"),
        "resources": state.get("resources", [])
    }
    
    messages = apply_prompt_template("reader", reader_input, configurable)
    
    # 使用vision模型
    llm = get_llm_by_type(AGENT_LLM_MAP["reader"])
    response = llm.invoke(messages)
    
    reader_response = response.content
    
    return Command(
        update={
            "reader_response": reader_response,
            "messages": [AIMessage(content=reader_response, name="reader")],
            "reader_request": None  # 清除请求
        },
        goto="analyzer"
    )

def thinker_node(
    state: State, config: RunnableConfig
) -> Command[Literal["router"]]:
    """思考器节点 - 使用推理模型，能够深度思考问题"""
    logger.info("Thinker node is processing complex reasoning task.")
    configurable = Configuration.from_runnable_config(config)
    
    current_plan = state.get("current_plan")
    current_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            current_step = step
            break
    
    if not current_step:
        return Command(goto="router")
    
    # 准备带有步骤记忆的输入
    memory_context = _prepare_step_memory(state, current_step)
    thinker_input = {
        "messages": [
            HumanMessage(
                content=f"{memory_context}# Current Task\n\n## Title\n{current_step.title}\n\n## Description\n{current_step.description}\n\n## Locale\n{state.get('locale', 'en-US')}"
            )
        ],
        "locale": state.get("locale", "en-US"),
        "resources": state.get("resources", [])
    }
    
    messages = apply_prompt_template("thinker", thinker_input, configurable)
    
    # 使用推理模型
    response = get_llm_by_type(AGENT_LLM_MAP["thinker"]).invoke(messages)
    
    current_step.execution_res = response.content
    
    # 更新步骤记忆
    step_memories = _update_step_memory(state, current_step.title, [response], 1)
    
    return Command(
        update={
            "messages": [AIMessage(content=response.content, name="thinker")],
            "step_memories": step_memories,
            "observations": state.get("observations", []) + [response.content]
        },
        goto="router"
    )

def reporter_node(state: State, config: RunnableConfig) -> dict:
    """报告生成节点 - 综合所有步骤记忆生成最终报告"""
    logger.info("Reporter node is generating comprehensive final report.")
    configurable = Configuration.from_runnable_config(config)
    
    current_plan = state.get("current_plan")
    step_memories = state.get("step_memories", {})
    
    # 构建综合报告输入
    comprehensive_context = f"""
# Task Overview

## Original Task
{current_plan.title}

## Task Description
{current_plan.thought}

# Comprehensive Step Results

"""
    
    # 添加所有步骤记忆
    for step_id, memory in step_memories.items():
        comprehensive_context += f"{memory}\n\n---\n\n"
    
    # 添加最终观察结果
    observations = state.get("observations", [])
    if observations:
        comprehensive_context += "# Final Observations\n\n"
        for i, obs in enumerate(observations, 1):
            comprehensive_context += f"## Observation {i}\n\n{obs}\n\n"
    
    # 构建reporter输入
    reporter_input = {
        "messages": [
            HumanMessage(
                content=f"{comprehensive_context}\n\n# Reporter Task\n\nPlease create a comprehensive final report that synthesizes all the above information. The report should be well-structured, clear, and actionable."
            )
        ],
        "locale": state.get("locale", "en-US"),
    }
    
    # 应用reporter模板
    invoke_messages = apply_prompt_template("reporter", reporter_input, configurable)
    
    # 生成最终报告
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
    logger.info("Comprehensive final report generated successfully.")
    
    return {"final_report": response.content}

async def _execute_agent_step(
    state: State, 
    config: RunnableConfig, 
    agent_type: str, 
    tools: list
) -> Command[Literal["router"]]:
    """执行agent步骤的通用函数（带记忆管理）"""
    configurable = Configuration.from_runnable_config(config)
    current_plan = state.get("current_plan")
    
    # 找到当前未执行的步骤
    current_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            current_step = step
            break
    
    if not current_step:
        logger.warning("No unexecuted step found")
        return Command(goto="router")
    
    logger.info(f"Executing step: {current_step.title} with agent: {agent_type}")
    
    # 准备带有步骤记忆的输入
    memory_context = _prepare_step_memory(state, current_step)
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"{memory_context}# Current Task\n\n## Title\n{current_step.title}\n\n## Description\n{current_step.description}\n\n## Locale\n{state.get('locale', 'en-US')}"
            )
        ],
        "locale": state.get("locale", "en-US"),
        "resources": state.get("resources", [])
    }
    
    messages = apply_prompt_template(agent_type, agent_input, configurable)
    
    # 处理MCP服务器配置
    mcp_servers = {}
    if configurable.mcp_settings:
        for server_name, server_config in configurable.mcp_settings["servers"].items():
            if (
                server_config["enabled_tools"]
                and agent_type in server_config["add_to_agents"]
            ):
                mcp_servers[server_name] = {
                    k: v for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env")
                }
    
    # 创建并执行agent
    if mcp_servers:
        async with  MultiServerMCPClient_wFileUpload(mcp_servers, state) as client:
            loaded_tools = tools[:]
            for tool in client.get_tools():
                loaded_tools.append(tool)
            agent = create_agent(agent_type, agent_type, loaded_tools, agent_type)
    else:
        agent = create_agent(agent_type, agent_type, tools, agent_type)
    
    # 执行agent
    recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", "25"))
    result = await agent.ainvoke(
        input={"messages": messages}, 
        config={"recursion_limit": recursion_limit}
    )
    
    # 处理结果
    response_content = result["messages"][-1].content
    current_step.execution_res = response_content
    
    # 估算交互次数并更新步骤记忆
    interaction_count = len(result.get("messages", []))
    step_memories = _update_step_memory(state, current_step.title, result.get("messages", []), interaction_count)
    
    logger.info(f"Step '{current_step.title}' completed by {agent_type}")
    
    return Command(
        update={
            "messages": [HumanMessage(content=response_content, name=agent_type)],
            "step_memories": step_memories,
            "observations": state.get("observations", []) + [response_content],
        },
        goto="router",
    )

def _analyze_file_types(resources: list) -> dict:
    """分析资源文件类型"""
    data_files = []
    pdf_image_files = []
    
    for resource in resources:
        file_name = resource.title.lower()
        if any(ext in file_name for ext in ['.csv', '.xlsx', '.xls', '.json', '.parquet']):
            data_files.append(resource)
        elif any(ext in file_name for ext in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp']):
            pdf_image_files.append(resource)
    
    return {
        "data_files": data_files,
        "pdf_image_files": pdf_image_files
    }