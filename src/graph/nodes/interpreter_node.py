from .base_node import BaseNode
from src.config.configuration import Configuration
from src.prompts.template import apply_prompt_template
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from typing import Literal, Dict, Any


class InterpreterNode(BaseNode):
    """编程节点 - 处理编程任务"""
    
    def __init__(self, model, tool_manager):
        redirect_tools = [
            {
                "name": "display_result",
                "description": "This function used to display your result to Supervisor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "object",
                            "properties": {
                                "generate": {
                                    "type": "string",
                                    "description": "Generated analysis report content with markdown formatting and download links."
                                },
                                "execution": {
                                    "type": "string", 
                                    "description": "Execution log showing chart generation status and download links."
                                }
                            },
                            "required": ["generate", "execution"],
                            "description": "The analysis result containing generated content and execution log."
                        },
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The display name of the generated file."
                                    },
                                    "path": {
                                        "type": "string",
                                        "description": "The full path to the generated file (e.g., sandbox:/mnt/data/filename.png)."
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "The file type/extension (e.g., png, jpg, pdf, etc.)."
                                    }
                                },
                                "required": ["name", "path", "type"]
                            },
                            "description": "List of generated files with their metadata."
                        }
                    },
                    "required": ["result"]
                }
            }
        ]
        tools = redirect_tools + \
            tool_manager.get_tools_for_node("interpreter")
        super().__init__("interpreter", model.bind_tools(tools), tool_manager)

    def __call__(self, state: Dict[str, Any], config: RunnableConfig) -> Command[Literal["supervisor"]]:
        
        configurable = Configuration.from_runnable_config(config)
        messages = apply_prompt_template(
            self.name, state, configurable, state["current_plan"],
            state["current_step_index"])
        response = self.model.invoke(messages, config)
        response.name = self.name

        # Must return with tool calls
        assert hasattr(response, "tool_calls") and response.tool_calls

        redirect_tool_calls = [
            i for i in response.tool_calls if i["name"] == "display_result"
        ]
        response.additional_kwargs["tool_calls"] = response.tool_calls = [
            i for i in response.tool_calls if i["name"] != "display_result"
        ]

        if len(redirect_tool_calls) > 0:
            # action is done and ready for review
            response.content = redirect_tool_calls[0]["args"]["result"]
            return Command(
                update={
                    "messages": [response],
                    "supervisor_iterate_time": state.get("supervisor_iterate_time", 0) + 1,
                    "tool_call_iterate_time" : 0
                },
                goto="supervisor"
            )
        elif len(response.tool_calls) > 0:
            # trigger the tool call
            tool_call_iterate_time = state.get("tool_call_iterate_time", 0)
            assert (
                tool_call_iterate_time <
                configurable.max_toolcall_iterate_times
            )
            return Command(
                update={
                    "messages": [response],
                    "tool_call_iterate_time": tool_call_iterate_time + 1
                },
                goto="{}_tools".format(self.name)
            )
