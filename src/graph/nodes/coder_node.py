from src.llms.llm import get_llm_by_type
from .base_node import BaseNode
from src.config.agents import AgentConfiguration
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.config.configuration import Configuration
from src.prompts.template import apply_prompt_template
from src.llms.llm import get_llm_by_type
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from typing import Literal, Dict, Any

from src.tools.llm_sandbox.langchain_tools import toolkit


class CoderNode(BaseNode):

    def __init__(self, toolmanager):
        super().__init__(
            "coder", AgentConfiguration.NODE_CONFIGS["writer"], toolmanager
        )
        # 输出给superviser的参数
        self.call_supervisor = {
            "name": "display_result",
            "description": "This function used to display your result to Supervisor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "codes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The code content.",
                                },
                                "file_name": {
                                    "type": "string",
                                    "description": "The name of the file.",
                                },
                            },
                            "required": ["content", "file_name"],
                        },
                        "description": "The list of coder files.",
                    }
                },
                "required": ["codes"],
            },
        }

        self.python_repl_tool = {
            "name": "python_repl_tool",
            "description": "Use this to execute python code and do data analysis or calculation. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute to do further analysis or calculation.",
                    }
                },
                "required": ["code"],
            },
        }

    def get_sandbox_tools(self):
        # Get bunch of sdk tools
        self.sdk_tools = toolkit.get_tools()
        self.llm_sandbox_execute_code_tool = None
        for tool in self.sdk_tools:
            if tool.name == "execute_python_code_sdk":
                self.llm_sandbox_execute_code_tool = tool
                break
        # if self.llm_sandbox_execute_code_tool is not None:
        #     from langchain_core.utils.function_calling import convert_to_openai_tool

        #     self.llm_sandbox_execute_code_tool_template = convert_to_openai_tool(
        #         self.llm_sandbox_execute_code_tool
        #     )
        #     self.log_execution("Prepared execute_python_code_sdk tool!!!")

        # else:
        #     self.llm_sandbox_execute_code_tool_template = None
        #     self.log_earning(
        #         "No execute_python_code_sdk tool found, using python_repl_tool instead"
        #     )
        self.llm_sandbox_execute_code_tool_template = {
            "name": "execute_python_code_sdk",
            "description": "Use this to execute python code and do data analysis or calculation. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute to do further analysis or calculation.",
                    }
                },
                "required": ["code"],
            },
        }
        # self.llm_sandbox_execute_code_tool_template = None

    async def execute(
        self, state: Dict[str, Any], config: RunnableConfig
    ) -> Command[Literal["supervisor"]]:

        configurable = Configuration.from_runnable_config(config)
        self.get_sandbox_tools()
        supervisor_iterate_time = state["supervisor_iterate_time"]
        messages = apply_prompt_template("coder", state, configurable)

        tools = [self.call_supervisor]
        if self.llm_sandbox_execute_code_tool_template is not None:
            tools.append(self.llm_sandbox_execute_code_tool_template)
        else:
            tools.append(self.python_repl_tool)
            self.log_execution_warning(
                "No execute_python_code_sdk tool found, using python_repl_tool instead"
            )
        print("=" * 80)
        print("Registered tools: {}".format(tools))
        print("=" * 80)
        self.log_input_message(messages)
        llm = get_llm_by_type(self.config.llm_type).bind_tools(tools)
        response = llm.invoke(messages)
        print("Coder Node response:")
        print(response)
        print("=" * 80)

        node_res_summary = ""
        iterate_times = state.get("tool_call_iterate_time", 0)
        if hasattr(response, "tool_calls") and response.tool_calls:
            iterate_times += 1
            self.log_tool_call(response, iterate_times)
            for tool_call in response.tool_calls:
                if tool_call["name"] == "display_result":
                    node_res_summary += f"\n{tool_call['args']['codes']}"
                    return Command(
                        update={
                            "messages": [
                                response,
                                HumanMessage(content=node_res_summary, name="coder"),
                            ],
                            "tool_call_iterate_time": 0,
                            "supervisor_iterate_time": supervisor_iterate_time + 1,
                        },
                        goto="supervisor",
                    )
                elif tool_call["name"] == "execute_python_code_sdk":
                    code = tool_call["args"]["code"]
                    libraries = tool_call["args"].get("libraries", [])
                    result = await toolkit.client.run_code(
                        code=code,
                        libraries=libraries,
                        session_id=state["llm_sandbox_session_id"],
                    )
                    if result.return_code == 0:
                        ci_result = f"Code executed successfully:\n```python\n{code}\n```\nResult: {result}"
                        self.log_execution(ci_result)
                    else:
                        ci_result = f"Error executing code:\n```python\n{code}\n```\nError: {result}"

                    return Command(
                        update={
                            "messages": [
                                response,
                                ToolMessage(
                                    content=ci_result,
                                    tool_call_id=tool_call["id"],
                                ),
                            ],
                            "tool_call_iterate_time": iterate_times,
                        },
                        goto="coder",
                    )

                elif tool_call["name"] == "python_repl_tool":
                    code = tool_call["args"]["code"]
                    ci_result = ""
                    from langchain_experimental.utilities import PythonREPL

                    repl = PythonREPL()
                    try:
                        result = repl.run(code)
                        # Check if the result is an error message by looking for typical error patterns
                        if isinstance(result, str) and (
                            "Error" in result or "Exception" in result
                        ):
                            self.log_execution_error(result)
                            ci_result = f"Error executing code:\n```python\n{code}\n```\nError: {result}"
                        else:
                            ci_result = f"Code executed successfully:\n```python\n{code}\n```\nResult: {result}"
                        self.log_execution(f"Code execution successful, {ci_result}")
                    except BaseException as e:
                        error_msg = repr(e)
                        self.log_execution_error(error_msg)
                        ci_result = f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"

                    return Command(
                        update={
                            "messages": [
                                response,
                                ToolMessage(
                                    content=ci_result,
                                    tool_call_id=tool_call["id"],
                                ),
                            ],
                            "tool_call_iterate_time": iterate_times,
                        },
                        goto="coder",
                    )

        else:
            self.log_execution_error("no tool call")
            # return Command(
            #     update={
            #         "messages": [
            #             AIMessage(
            #                 content="No tool call found, please check your code or tools."
            #             )
            #         ],
            #         "tool_call_iterate_time": iterate_times,
            #     },
            #     goto="coder",
            # )
            raise ValueError
