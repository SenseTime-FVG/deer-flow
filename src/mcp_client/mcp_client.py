from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import NonTextContent
from src.graph.types import State, Resource
from typing import Any
import os.path as osp
import uuid
import json
import re
import base64
import filetype
import logging
import traceback
logger = logging.getLogger(__name__)

from src.utils.file_utils import file_to_data_uri, base64_to_bytes


class MultiServerMCPClient_wFileUpload(MultiServerMCPClient):
    def __init__(self, connections, state: State):
        super().__init__(connections)
        self.state = state

    async def _initialize_session_and_load_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Initialize a session and load tools from it.

        Args:
            server_name: Name to identify this server connection
            session: The ClientSession to initialize
        """
        # Initialize the session
        await session.initialize()
        self.sessions[server_name] = session

        # Load tools from this server
        server_tools = await load_mcp_tools(session)
        server_tools = [self._add_file_upload_in_tool(tool) for tool in server_tools]
        self.server_name_to_tools[server_name] = server_tools
    
    # def _add_file_upload_in_tool(self,tool: BaseTool):
    #     old_coroutine = tool.coroutine

    #     def _recursive_replace_uri_with_file(out):
    #         logger.info(f"recursive_replace_uri_with_file before load, {type(out)}")
    #         if isinstance(out, str):
    #             try:
    #                 logger.info(f"recursive_replace_uri_with_file in json.load")
    #                 parsed = json.loads(out)
    #                 processed = _recursive_replace_uri_with_file(parsed)
    #                 return json.dumps(processed, ensure_ascii=False)
    #             except Exception as e:
    #                 logger.info(f"recursive_replace_uri_with_file in json.load error, {e}")

    #             if out.startswith('data:') and 'base64' in out:
    #                 logger.info(f"recursive_replace_uri_with_file processing base64 content")
    #                 out_bytes = base64_to_bytes(out)
    #                 file_name = f"{uuid.uuid4().hex[:8]}.{filetype.guess(out_bytes).extension}"
    #                 file_path = osp.join(self.state['session_dir'], file_name)
    #                 with open(file_path, 'wb') as file:
    #                     file.write(out_bytes)
    #                 self.state['resources'].append(Resource(
    #                     uri=file_path,
    #                     title=file_path,
    #                     description=file_path))
    #                 return file_path
    #             else:
    #                 return out
    #         elif isinstance(out, dict):
    #             for k, v in out.items():
    #                 out[k] = _recursive_replace_uri_with_file(v)
    #         elif isinstance(out, list):
    #             out = [_recursive_replace_uri_with_file(item) for item in out]
    #         elif isinstance(out, tuple):
    #             out = tuple([_recursive_replace_uri_with_file(item) for item in out])
    #         elif out is None:
    #             pass
    #         else:
    #             logger.error(f"recursive_replace_uri_with_file error, {type(out)}")
    #             raise ValueError(f"recursive_replace_uri_with_file error, {type(out)}")
    #         return out
    def _add_file_upload_in_tool(self, tool: BaseTool):
        old_coroutine = tool.coroutine

        def _replace_base64_with_path(out):
            # todo: 有潜在bug，如果模型输出内容就是有base64编码的内容，则会被替换为文件路径
            # 需要针对特定返回格式的json string进行处理
            # todo: 如果返回内容中有两个相同的文件？
            if isinstance(out, str):
                # 用正则表达式匹配out中的base64编码的uri，并替换为文件路径
                pattern = r'data:[^;]+;base64,[A-Za-z0-9+/=]+'
                matches = re.findall(pattern, out)
                for match in matches:
                    b64_string = match
                    # 使用filetype库猜测文件类型，并使用uuid生成文件名
                    data_bytes = base64_to_bytes(b64_string)
                    file_name = f"{uuid.uuid4().hex[:8]}.{filetype.guess(data_bytes).extension}"
                    file_path = osp.join(self.state['session_dir'], file_name)
                    with open(file_path, 'wb') as file:
                        file.write(data_bytes)
                    self.state['resources'].append(Resource(
                        uri=file_path,
                        title=file_path,
                        description=file_path))
                    out = out.replace(b64_string, file_path)
                return out
            elif isinstance(out, dict):
                for k, v in out.items():
                    out[k] = _replace_base64_with_path(v)
                return out
            elif isinstance(out, list):
                return [_replace_base64_with_path(item) for item in out]
            elif isinstance(out, tuple):
                return tuple([_replace_base64_with_path(item) for item in out])
            elif out is None:
                return out
            else:
                logger.error(f"recursive_replace_uri_with_file error, {type(out)}")
                raise ValueError(f"recursive_replace_uri_with_file error, {type(out)}")
        
        async def wrapped_call_tool(
            **arguments: dict[str, Any],
        ) -> tuple[str | list[str], list[NonTextContent] | None]:
            for k in list(arguments.keys()):
                if 'uri' == k.lower() and osp.exists(arguments[k]):
                    arguments[k] = file_to_data_uri(arguments[k])
            out = await old_coroutine(**arguments)
            try:
                out = _replace_base64_with_path(out)
            except Exception as e:
                logger.error(f"recursive_replace_uri_with_file error, {e}")
                logger.error(traceback.format_exc())
                raise e
            logger.info(f"wrapped_call_tool out: {out}")
            return out
        
        tool.coroutine = wrapped_call_tool
        return tool
