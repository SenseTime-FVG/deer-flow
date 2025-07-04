import json
from langchain_core.messages import HumanMessage, ToolMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
import random
from src.config.configuration import Configuration
from src.graph.nodes.base_node import BaseNode
from src.prompts.planner_model import TaskStatus
from src.prompts.template import apply_prompt_template
from typing import Literal, Dict, Any


class SupervisorNode(BaseNode):
    """Supervisor节点 - 评估步骤完成度"""

    def __init__(
        self, model, tool_manager, messages_key: str = "supervisor_message"
    ):
        redirect_tools = [
            {
                "name": "advise",
                "description": "This function is used to send advice to the action worker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "suggestion": {
                            "type": "string",
                            "description": "Identify specific issues using professional terminology, and clearly articulate the direction for improvement."
                        },
                        "score": {
                            "type": "number",
                            "description": "Score for the action result."
                        }
                    },
                    "required": ["suggestion", "score"]
                }
            },
            {
                "name": "replan",
                "description": "If continuing to execute the current plan based on existing information cannot resolve the user's problem, send a call signal to the function to trigger a plan update.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "The reason to replan."
                        }
                    },
                    "required": ["reason"]
                }
            },
            {
                "name": "complete",
                "description": "This function sends a signal to the manager to change the action status to 'complete'. This function will not get a response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_id": {
                            "type": "string",
                            "description": "The action ID."
                        }
                    },
                    "required": ["action_id"]
                }
            }
        ]
        super().__init__(
            "supervisor", model.bind_tools(redirect_tools), tool_manager,
            messages_key)

    def __call__(self, state: Dict[str, Any], config: RunnableConfig) \
            -> Command[Literal["writer", "reporter", "searcher", "coder", "interpreter", "reader", "receiver", "__end__"]] | Dict[str, Any]:
        """执行supervisor逻辑"""

        configurable = Configuration.from_runnable_config(config)
        messages = [apply_prompt_template(self.name, state, configurable)] + \
            state[self.messages_key]
        response = self.model.invoke(messages, config)
        response.name = self.name
        self.log_execution(response)

        # default to complete
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            response.tool_calls = [{
                "name": "complete",
                "args": {"action_id": state["current_step_index"]},
                "id": "toolu_vrtx_{}".format(random.randint(0, 2 ** 24)),
                "type": "tool_call"
            }]

        assert len(response.tool_calls) == 1
        redirect = response.tool_calls[0]
        action = self.get_action(
            state["current_plan"], state["current_step_index"])

        # max_supervisor_iterate_times = configurable.max_supervisor_iterate_times
        message_to_review = state["action_message"][-1]
        if redirect["name"] == "advise":  # 打回重跑
            self.log_execution(f"Step {action.id} not complete")
            return Command(
                update={
                    self.messages_key: [
                        response,
                        ToolMessage(
                            "", tool_call_id=response.tool_calls[0]["id"])
                    ],
                    "action_message": [
                        ToolMessage(
                            redirect["args"]["suggestion"],
                            tool_call_id=message_to_review.tool_calls[0]["id"])
                    ],
                    "supervisor_iterate_time": state.get("supervisor_iterate_time", 0) + 1
                },
                goto=action.type.lower()
            )

        elif redirect["name"] == "replan":  # 重新规划
            content = "{}\n# 重新规划\n根据已有信息，需要更新 plan 中 {} 以及之后的各 actions 和 goals （已经运行过的 actions 不再更改），才能更好解决用户问题。".format(
                redirect["args"]["reason"], action.id)
            return Command(
                update={
                    self.messages_key: [
                        response,
                        ToolMessage(
                            "", tool_call_id=response.tool_calls[0]["id"])],
                    "action_message": [
                        ToolMessage(
                            "已重新规划任务",
                            tool_call_id=message_to_review.tool_calls[0]["id"])
                    ],
                    "plan_messages": [
                        HumanMessage(content=content, name=self.name)
                    ],
                    "current_step_index": action.id,
                    "supervisor_iterate_time": 0,
                },
                goto="planner"
            )

        elif redirect["name"] == "complete":
            action.status = TaskStatus.COMPLETED
            action.result = json.dumps(
                message_to_review.tool_calls[0]["args"], ensure_ascii=False)
            next_action = self.get_next_action(
                state["current_plan"], state["current_step_index"])
            if next_action is None:  # 全部任务完成，汇总信息返回
                self.log_execution(f"Plan complete")
                return Command(
                    update={
                        self.messages_key: [
                            response,
                            ToolMessage(
                                "", tool_call_id=response.tool_calls[0]["id"])
                        ],
                        "default_action_messages": {
                            action.id.lower(): {
                                self.messages_key: messages + [response]
                            },
                        },
                        "final_report": action.result,
                    },
                    goto="__end__"
                )

            else:  # 任务完成继续任务
                self.log_execution(
                    f"Step {action.id} completed and go to {next_action.id}")
                message_to_action_workers = \
                    self.get_action_with_dependencies_json(
                        state["current_plan"], state["current_step_index"],
                        state.get("resources", []))
                return Command(
                    update={
                        self.messages_key: [
                            response,
                            RemoveMessage(id="__remove_all__")
                        ],
                        "action_message": [
                            RemoveMessage(id="__remove_all__"),
                            HumanMessage(
                                message_to_action_workers, name=self.name)
                        ],
                        "default_action_messages": {
                            action.id.lower(): {
                                self.messages_key: messages + [response]
                            },
                        },
                        "current_step_index": next_action.id,
                        "supervisor_iterate_time": 0,
                    },
                    goto=next_action.type.lower()
                )

        else:
            raise Exception("Incorrect redirection")
