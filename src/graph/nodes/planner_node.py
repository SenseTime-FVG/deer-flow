# nodes/planner_node.py
"""规划器节点"""

from .base_node import BaseNode
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from typing import Literal, Dict, Any
import json
from src.config.configuration import Configuration
from src.prompts.template import apply_prompt_template
from src.prompts.planner_model import Plan


class PlannerNode(BaseNode):
    """规划器节点 - 制定执行计划"""

    def __init__(
        self, model, tool_manager, messages_key: str = "plan_messages"
    ):
        super().__init__("planner", model, tool_manager, messages_key)

    def __call__(self, state: Dict[str, Any], config: RunnableConfig) \
            -> Command[Literal["writer", "coder", "interpreter", "searcher", "reader", "reporter", "receiver", "__end__"]]:

        configurable = Configuration.from_runnable_config(config)
        messages = [apply_prompt_template(self.name, state, configurable)] + \
            state[self.messages_key]
        response = self.model.invoke(messages, config)
        response.name = self.name

        plan_content = response.content.split(
            "<|plan|>")[1].split("<|end|>")[0]
        # plan_content = response.model_dump_json(indent=4, exclude_none=True)
        """ plan 输出结果示意
    [
        {
            "title": "AI Market Research Project",
            "description": "Comprehensive analysis of current AI market trends and opportunities",
            "goals": [
                {
                    "id": "G1",
                    "description": "Market data collection and analysis",
                    "actions": [
                        {
                            "id": "G1-A1",
                            "description": "Search for current AI market size and growth data",
                            "type": "searcher",
                            "dependencies": [],
                            "details": "Focus on 2024-2025 market data from reliable sources",
                            "references": [],
                            "status": "pending"
                        },
                        {
                            "id": "G1-A2", 
                            "description": "Analyze collected market data and identify trends",
                            "type": "interpreter",
                            "dependencies": ["G1-A1"],
                            "details": "Create visual representations and key insights",
                            "references": [],
                            "status": "pending"
                        }
                    ]
                },
                {
                    "id": "G2",
                    "description": "Report generation",
                    "actions": [
                        {
                            "id": "G2-A1",
                            "description": "Generate comprehensive market research report",
                            "type": "reporter", 
                            "dependencies": ["G1-A2"],
                            "details": "Include executive summary, detailed analysis, and recommendations",
                            "references": [],
                            "status": "pending"
                        }
                    ]
                }
            ]
        }
    ]
        """
        self.log_execution(f"Generated plan: {plan_content}")

        plan_dict = json.loads(plan_content)
        plan = Plan.model_validate(plan_dict)
        actions = [
            action for goal in plan.goals
            for action in goal.actions
        ]

        assert len(actions) > 0
        current_step_index = (
            state["current_step_index"]  # updated plan
            if "current_step_index" in state
            else actions[0].id  # new plan
        )
        message_to_action_workers = self.get_action_with_dependencies_json(
            plan, current_step_index, state.get("resources", []))
        return Command(
            update={
                self.messages_key: [response],
                "default_{}".format(self.messages_key): (
                    messages + [response]
                ),
                "action_message": [
                    HumanMessage(message_to_action_workers, name=self.name)
                ],
                "current_plan": plan,
                "current_step_index": current_step_index,
            },
            goto=self.get_action(plan, current_step_index).type.lower()
        )
