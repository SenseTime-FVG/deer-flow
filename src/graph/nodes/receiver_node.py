from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
import random
from src.config.configuration import Configuration
from src.graph.nodes.base_node import BaseNode
from src.prompts.planner_model import TaskStatus
from src.prompts.template import apply_prompt_template
from typing import Any, Dict, Literal


class ReceiverNode(BaseNode):

    def __init__(
        self, model, tool_manager, messages_key: str = "action_message"
    ):
        redirect_tools = [
            {
                "name": "display_result",
                "description": "This function is used to display your result to the user and Supervisor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "A comprehensive markdown-formatted text that consolidates all user-provided information in a clear and organized format."
                        },
                        "references": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Unique identifier for the reference."},
                                    "type": {"type": "string", "description": "Type of reference (e.g., file, image)."},
                                    "name": {"type": "string", "description": "File name."},
                                    "function": {"type": "string", "description": "Description of the file function."}
                                },
                                "required": ["id", "type", "name", "function"]
                            },
                            "description": "List of reference materials provided by the user."
                        }
                    },
                    "required": ["result"]
                }
            }
        ]
        tools = redirect_tools + tool_manager.get_tools_for_node("receiver")
        super().__init__(
            "receiver", model.bind_tools(tools), tool_manager, messages_key)
        self.redirect_tool_names = [i["name"] for i in redirect_tools]

    def __call__(self, state: Dict[str, Any], config: RunnableConfig) -> Command[Literal["supervisor"]]:

        configurable = Configuration.from_runnable_config(config)
        messages = [apply_prompt_template(self.name, state, configurable)] + \
            state[self.messages_key]
        response = self.model.invoke(messages, config)
        response.name = self.name

        # default to display_result
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            response.tool_calls = [{
                "name": "display_result",
                "args": {"result": response.content},
                "id": "toolu_vrtx_{}".format(random.randint(0, 2 ** 24)),
                "type": "tool_call"
            }]

        # either multiple regular tool calls or single redirect tool call
        redirect_tool_calls = [
            i for i in response.tool_calls
            if i["name"] in self.redirect_tool_names
        ]
        assert len(redirect_tool_calls) == 0 or (
            len(redirect_tool_calls) == 1 and len(response.tool_calls) == 1
        )

        action = self.get_action(
            state["current_plan"], state["current_step_index"])
        action.status = TaskStatus.PROCESSING

        if len(redirect_tool_calls) > 0:
            # action is done and ready for review
            message_to_supervisor = "{}\n{}".format(
                self.get_action_with_dependencies_json(
                    state["current_plan"], state["current_step_index"],
                    state.get("resources", [])),
                redirect_tool_calls[0]["args"]["result"]
            )
            return Command(
                update={
                    self.messages_key: [response],
                    "supervisor_message": [
                        HumanMessage(message_to_supervisor, name=self.name)
                    ],
                    "default_action_messages": {
                        action.id.lower(): {
                            self.messages_key: messages + [response]
                        },
                    },
                    "supervisor_iterate_time": state.get("supervisor_iterate_time", 0) + 1,
                    "tool_call_iterate_time": 0
                },
                goto="supervisor"
            )
        elif len(response.tool_calls) > 0:
            # trigger the tool call
            tool_call_iterate_time = state.get("tool_call_iterate_time", 0)
            return Command(
                update={
                    self.messages_key: [response],
                    "tool_call_iterate_time": tool_call_iterate_time + 1
                },
                goto="{}_tools".format(self.name)
            )
