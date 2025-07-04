import aiohttp
import json
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
import os
import random
import re
from src.config.configuration import Configuration
from src.graph.nodes.base_node import BaseNode
from src.prompts.planner_model import TaskStatus
from src.prompts.template import apply_prompt_template
from typing import Literal, Dict, Any, List, Tuple
import uuid


class SearcherNode(BaseNode):

    def __init__(
        self, model, tool_manager, messages_key: str = "action_message"
    ):
        redirect_tools = [
            {
                "name": "display_result",
                "description": "This function used to display your result to Supervisor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "The list of search queries that were used to gather the information being summarized."
                        },
                        "result": {
                            "type": "string",
                            "description": "A comprehensive markdown-formatted summary of the search results, including key findings, structured information, and relevant details organized in a readable format."
                        }
                    },
                    "required": ["queries", "result"]
                }
            }
        ]
        tools = redirect_tools + tool_manager.get_tools_for_node("searcher")
        super().__init__(
            "searcher", model.bind_tools(tools), tool_manager, messages_key)
        self.redirect_tool_names = [i["name"] for i in redirect_tools]

    def __call__(self, state: Dict[str, Any], config: RunnableConfig) -> Command[Literal["supervisor"]]:

        configurable = Configuration.from_runnable_config(config)
        messages = [apply_prompt_template(self.name, state, configurable)] + \
            state[self.messages_key]
        self.log_input_message(messages)
        response = self.model.invoke(messages, config)
        response.name = self.name

        # default to display_result
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            response.tool_calls = [{
                "name": "display_result",
                "args": {"queries": [], "result": response.content},
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

    def _extract_markdown_images(self, text: str) -> List[Tuple[str, str]]:
        """提取 Markdown 格式图像的描述和 URL"""
        if not text:
            return []
        
        # 匹配 ![description](url) 格式
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(pattern, text)
        
        # 过滤并返回有效的 HTTP(S) URL 和对应的描述
        result = []
        for description, url in matches:
            url = url.strip()
            if url.startswith(('http://', 'https://')):
                result.append((description, url))
        
        return result
    
    async def _download_images_batch(
        self, 
        image_infos: List[Tuple[str, str]], 
        session_dir: str,
        max_images: int = 5,
        timeout: int = 10
    ) -> List[Dict[str, Any]]:
        """简化的批量下载"""
        
        os.makedirs(session_dir, exist_ok=True)
        downloaded_images = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            for i, (image_desc, url) in enumerate(image_infos[:max_images]):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            
                            # 简单的文件扩展名检测
                            if 'image/png' in response.headers.get('Content-Type', ''):
                                ext = '.png'
                            elif 'image/gif' in response.headers.get('Content-Type', ''):
                                ext = '.gif'
                            else:
                                ext = '.jpg'
                            
                            filename = f"research_img_{i+1}_{uuid.uuid4().hex[:8]}{ext}"
                            local_path = os.path.join(session_dir, filename)
                            
                            with open(local_path, 'wb') as f:
                                f.write(content)
                            
                            downloaded_images.append({
                                'original_url': url,
                                'local_path': local_path,
                                'image_desc': image_desc,
                                'mime_type': response.headers.get('Content-Type', 'image/jpeg'),
                                'size': len(content)
                            })
                            
                except Exception as e:
                    self.log_execution_warning(f"Failed to download {url}: {e}")
                    continue
        
        return downloaded_images
    
    async def _process_images(self, research_content: str, state: Dict[str, Any]) -> str:
        """处理图像下载和资源添加"""
        from src.graph.types import Resource
        
        # 提取图像URL
        image_listtuple = self._extract_markdown_images(research_content)
        
        if image_listtuple:
            self.log_execution(f"Found {len(image_listtuple)} images, downloading...")
            
            downloaded_images = await self._download_images_batch(
                image_listtuple[:5],  # 限制最多5张图
                session_dir=state.get('session_dir', './sessions/default'),
                max_images=5
            )
            
            new_images_info = []
            # 添加到resources
            for image_info in downloaded_images:
                new_images_info.append({
                    'uri': image_info['local_path'],
                    'title': image_info['image_desc'],
                    'description': image_info['image_desc'],
                })
                
                # 添加到state的resources中
                if 'resources' not in state:
                    state['resources'] = []
                    
                state['resources'].append(
                    Resource(
                        uri=image_info['local_path'],
                        title=image_info['image_desc'],
                        description=image_info['image_desc'],
                    )
                )
            
            self.log_execution(f"Downloaded {len(downloaded_images)} images")
            
            # 将图像信息添加到研究结果中
            research_content += f"\n##related images\n{json.dumps(new_images_info, indent=2, ensure_ascii=False)}"
        
        return research_content