from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool


class MakeImageMessage(BaseTool):
    """Tool for make an image message from URL for multimodal model."""

    name: str = 'make_image_message'
    description: str = 'Convert the image URL link to image message.'

    def _run(self, url: str, run_manager=None):
        return BaseMessage([{
            'type': 'image_url',
            'image_url': {'url': url}
        }], type='tool')
