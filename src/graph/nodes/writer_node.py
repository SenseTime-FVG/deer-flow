from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
import random
from src.config.configuration import Configuration
from src.graph.nodes.base_node import BaseNode
from src.prompts.planner_model import TaskStatus
from src.prompts.template import apply_prompt_template
from typing import Literal, Dict, Any


class WriterNode(BaseNode):

    def __init__(
        self, model, tool_manager, messages_key: str = "action_message"
    ):
        redirect_tools = [
            {
                "name": "display_result",
                "description": "This function used to display your result to user and Supervisor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "A comprehensive markdown-formatted text content, including the generated or processed text organized in a readable format."
                        }
                    },
                    "required": ["result"]
                }
            }
        ]
        super().__init__(
            "writer", model.bind_tools(redirect_tools), tool_manager,
            messages_key)

    def __call__(self, state: Dict[str, Any], config: RunnableConfig) -> Command[Literal["supervisor"]]:

        configurable = Configuration.from_runnable_config(config)
        messages = [apply_prompt_template(self.name, state, configurable)] + \
            state[self.messages_key]
        response = self.model.invoke(messages, config)
        response.name = self.name
        self.log_execution(response)

        # default to display_result
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            response.tool_calls = [{
                "name": "display_result",
                "args": {"result": response.content},
                "id": "toolu_vrtx_{}".format(random.randint(0, 2 ** 24)),
                "type": "tool_call"
            }]

        assert len(response.tool_calls) == 1
        redirect = response.tool_calls[0]

        action = self.get_action(
            state["current_plan"], state["current_step_index"])
        action.status = TaskStatus.PROCESSING

        message_to_supervisor = "{}\n{}".format(
            self.get_action_with_dependencies_json(
                state["current_plan"], state["current_step_index"],
                state.get("resources", [])),
            redirect["args"]["result"]
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
                "supervisor_iterate_time": state.get("supervisor_iterate_time", 0) + 1
            },
            goto="supervisor"
        )
