import json
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from src.config.configuration import Configuration
from src.graph.nodes.base_node import BaseNode
from src.prompts.template import apply_prompt_template
from typing import Literal, Dict, Any


class CoordinatorNode(BaseNode):
    """协调器节点 - 过滤无意义问题"""

    def __init__(
        self, model, tool_manager, messages_key: str = "task_messages"
    ):
        redirect_tools = [
            {
                "name": "planner",
                "description": "Trigger the task planning process, automatically advancing the planning workflow. No direct response is returned as the user will see the results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "A concise and clear task title."
                        },
                        "description": {
                            "type": "string",
                            "description": "A brief description summarizing the task."
                        },
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of specific requirements for the task."
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of constraints to follow during task execution."
                        },
                        "references": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Unique identifier for the reference."},
                                    "type": {"type": "string", "description": "Type of reference: file|knowledge"},
                                    "name": {"type": "string", "description": "File name."},
                                    "function": {"type": "string", "description": "Summarize the purpose of the document; the planner will relay this information."}
                                },
                                "required": ["id", "type", "name", "function"]
                            },
                            "description": "List of reference materials provided by the user. References will be indicated using the <file>/<knowledge> tags. Do not fabricate information."
                        },
                        "expected_outcome": {
                            "type": "string",
                            "description": "The desired outcome after completing the task."
                        },
                        "status": {
                            "type": "string",
                            "enum": ["collecting", "clarifying", "thinking", "completed"],
                            "description": "Current status of the task, indicating progress."
                        },
                        "confidence": {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "type": "number",
                                    "description": "Confidence score (0.00 - 1.00)."
                                },
                                "missing_info": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Potential missing information items."
                                }
                            },
                            "description": "Task confidence score (0.00 - 1.00). Use the score to determine the task's readiness. Ensure the score reaches 0.9 or above before setting the status to 'complete'.",
                            "required": ["score", "missing_info"]
                        }
                    },
                    "required": ["title", "description", "requirements", "constraints", "references", "expected_outcome", "status", "confidence"]
                }
            }
        ]
        tools = redirect_tools + tool_manager.get_tools_for_node("coordinator")
        super().__init__(
            "coordinator", model.bind_tools(tools), tool_manager, messages_key)
        self.redirect_tool_names = [i["name"] for i in redirect_tools]

    def __call__(self, state: Dict[str, Any], config: RunnableConfig) -> Command[Literal["planner", "__end__"]]:

        configurable = Configuration.from_runnable_config(config)
        messages = [apply_prompt_template(self.name, state, configurable)] + \
            state[self.messages_key]
        response = self.model.invoke(messages, config)
        response.name = self.name
        self.log_execution(response)

        if not (hasattr(response, "tool_calls") and response.tool_calls):
            return Command(
                update={
                    self.messages_key: [response],
                    "final_report": response.content
                },
                goto="__end__"
            )

        # either multiple regular tool calls or single redirect tool call
        redirect_tool_calls = [
            i for i in response.tool_calls
            if i["name"] in self.redirect_tool_names
        ]
        assert len(redirect_tool_calls) == 0 or (
            len(redirect_tool_calls) == 1 and len(response.tool_calls) == 1
        )

        if len(redirect_tool_calls) > 0:
            tool_call = redirect_tool_calls[0]
            return Command(
                update={
                    self.messages_key: [response],
                    "default_{}".format(self.messages_key): (
                        messages + [response]
                    ),
                    "plan_messages": [
                        HumanMessage(
                            json.dumps(tool_call["args"], ensure_ascii=False),
                            name=self.name)
                    ]
                },
                goto="planner"
            )
        else:
            return Command(
                update={self.messages_key: [response]},
                goto="{}_tools".format(self.name)
            )
