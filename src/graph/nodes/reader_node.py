# nodes/reader_node.py
"""阅读器节点"""

import base64
import io
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
import os
from PIL import Image
from src.config.configuration import Configuration
from src.graph.nodes.base_node import BaseNode
from src.prompts.template import apply_prompt_template
from typing import Literal, Dict, Any
import uuid


class ReaderNode(BaseNode):
    """阅读器节点 - 处理图像理解任务"""
    
    def __init__(self, model, tool_manager):
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
            "reader", model.bind_tools(redirect_tools), tool_manager)

    def __call__(self, state: Dict[str, Any], config: RunnableConfig) -> Command[Literal["supervisor"]]:

        configurable = Configuration.from_runnable_config(config)
        messages = apply_prompt_template(
            self.name, state, configurable, state["current_plan"],
            state["current_step_index"])
        response = self.model.invoke(messages, config)
        response.name = self.name
        self.log_execution(response)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            redirect_tool_calls = [
                i for i in response.tool_calls if i["name"] == "display_result"
            ]
            response.additional_kwargs["tool_calls"] = response.tool_calls = []

            if len(redirect_tool_calls) > 0:
                response.content = redirect_tool_calls[0]["args"]["result"]

        return Command(
            update={
                "messages": [response],
                "supervisor_iterate_time": state.get("supervisor_iterate_time", 0) + 1,
            },
            goto="supervisor"
        )

    async def _handle_tool_calls(self, response, messages, llm, configurable, state, session_dir):
        """处理工具调用"""
        from src.tools.image_rotate import rotate_image
        from src.graph.types import Resource
        
        max_toolcall_iterate_times = configurable.max_toolcall_iterate_times
        iterate_times = 0
        
        while hasattr(response, 'tool_calls') and response.tool_calls and iterate_times < max_toolcall_iterate_times:
            iterate_times += 1
            self.log_execution(f"Reader tool call iteration: {iterate_times}")
            
            for tool_call in response.tool_calls:
                if tool_call["name"] == "call_rotate_tool":
                    image_path = tool_call["args"]["file_info"]
                    rotate_request = tool_call["args"]["rotate_request"]
                    
                    # 生成新的文件路径
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    extension = os.path.splitext(image_path)[1]
                    
                    # 使用UUID确保文件名唯一性
                    unique_id = str(uuid.uuid4())[:8]
                    new_filename = f"{base_name}_rotated_{unique_id}{extension}"
                    new_file_path = os.path.join(session_dir, new_filename)
                    
                    try:
                        # 旋转并保存图像
                        rotation_desc = rotate_image(image_path, new_file_path, rotate_request)
                        
                        # 更新资源列表
                        if 'resources' not in state:
                            state['resources'] = []
                            
                        state['resources'].append(Resource(
                            uri=new_file_path,
                            title=f"{base_name} ({rotation_desc})",
                            description=f"从 {image_path} {rotation_desc}后生成的图像"
                        ))
                        
                        self.log_execution(f"图像已成功{rotation_desc}并保存到: {new_file_path}")
                        
                        current_resources = f"""##Rotated image is: 
uri: {new_file_path}
title: {rotation_desc}
description: 从 {image_path} {rotation_desc}后生成的图像"""
                        
                        # 将结果返回给reader
                        mcp_summary = f"Rotate success! Tools Results:\n{current_resources}"
                        
                        # 第二次LLM调用：基于结果分析
                        messages = messages + [
                            AIMessage(content=response.content, tool_calls=response.tool_calls),
                            ToolMessage(content=mcp_summary, tool_call_id=response.tool_calls[0]["id"]),
                            self._create_message_with_base64_image(
                                text=f"Based on results, Continue to complete your task\n## Locale\n{state.get('locale', 'en-US')}",
                                image_paths=new_file_path
                            )
                        ]
                        
                        response = llm.invoke(messages)
                        
                    except Exception as e:
                        self.log_execution(f"Error rotating image: {e}")
                        # 继续处理，不中断流程
                        break
        
        return response