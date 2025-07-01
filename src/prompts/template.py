# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import dataclasses
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.prebuilt.chat_agent_executor import AgentState
from src.config.configuration import Configuration
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from pathlib import Path
# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_prompt_template(prompt_name: str) -> str:
    """
    Load and return a prompt template using Jinja2.

    Args:
        prompt_name: Name of the prompt template file (without .md extension)

    Returns:
        The template string with proper variable substitution syntax
    """
    try:
        template = env.get_template(f"{prompt_name}.md")
        return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")


def apply_prompt_template(
    prompt_name: str, state: AgentState, configurable: Configuration = None
) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute

    Returns:
        List of messages with the system prompt as the first message
    """
    # Convert state to dict for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
    }
    
    # Add configurable variables
    if configurable:
        state_vars.update(dataclasses.asdict(configurable))
    state_vars.update(state)
    try:
        template = env.get_template(f"{prompt_name}.md")
        system_prompt = template.render(**state_vars)
    
        return [SystemMessage(content=system_prompt)] + state["messages"]
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")


def simulate_user_template(
    prompt_name: str, state: AgentState, ai_content: str
) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute
        ai_content: the content of ai ask user

    Returns:
        List of messages with the system prompt as the first message
    """
   
    try:
        # 获取当前文件的 Path 对象
        current_file = Path(__file__).resolve()

        # 获取当前文件所在的目录
        current_dir = current_file.parent
        # 获取system prompt
        with open(os.path.join(current_dir, f"{prompt_name}.md"), "r", encoding="utf-8") as f:
            system_prompt = f.read()
    
        return [SystemMessage(content=system_prompt)] + state['messages'] + [HumanMessage(content=ai_content)]
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")