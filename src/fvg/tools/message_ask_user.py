from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing import Type


class MessageAskUserToolInput(BaseModel):
    text: str = Field(
        description='The question text for user interaction, structured clearly to guide the user\'s response.')


class MessageAskUserTool(BaseTool):
    """The tool to ask and collect information from the user."""

    name: str = 'message_ask_user'
    description: str = 'Ask user a question and wait for response. Use for requesting clarification, asking for confirmation, or gathering additional information.'
    args_schema: Type[BaseModel] = MessageAskUserToolInput

    def _run(self, text: str, run_manager=None):
        feedback = interrupt(text)
        return str(feedback)
