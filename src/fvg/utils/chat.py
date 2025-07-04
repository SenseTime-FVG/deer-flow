import argparse
import base64
import filetype
import fvg.common
from fvg.messages.utils import convert_to_openai_messages
import json
from langchain_core.messages import AIMessage
from langgraph.types import Command
import urllib.request


class ChuckPrinter:
    def __init__(self):
        self.current_step = 0

    def update(self, message_chunk, metadata):
        if self.current_step != metadata['langgraph_step']:
            if self.current_step > 0:
                print()

            self.current_step = metadata['langgraph_step']
            print('{}:'.format(metadata['langgraph_node']), end='')
            if (
                metadata['langgraph_node'] == 'model' and
                'tags' in metadata and metadata['tags'] is not None
            ):
                for i in metadata['tags']:
                    print('{}:'.format(i), end='')

        if message_chunk.content:
            print(message_chunk.content, end='', flush=True)

        if hasattr(message_chunk, 'tool_calls'):
            for i in message_chunk.tool_calls:
                if i['name']:
                    print('\n->{}:'.format(i['name']), end='')

                if i['args']:
                    print(i['args'], end='')


def print_message(node: str, response: dict):
    for message in response['messages']:
        print('{}:{}'.format(node, message.content))
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_content = '->{}: {}'.format(
                    tool_call['name'], tool_call['args'])
                print(tool_call_content)


def make_user_message_content_list(query_list: str):
    content_list = []
    complex_content = None
    supported_proxies = ['file:', 'data:', 'http:', 'https:']
    for i in query_list:
        if any([i.startswith(j) for j in supported_proxies]):
            # supported URI
            with urllib.request.urlopen(i) as s:
                data = s.read()

            if complex_content is None:
                complex_content = []

            mime_type = filetype.guess_mime(data)
            base64_data = base64.b64encode(data).decode('utf-8')
            if mime_type.startswith('image'):
                data_uri = 'data:{};base64,{}'.format(mime_type, base64_data)
                complex_content.append({
                    'type': 'image_url',
                    'image_url': {'url': data_uri},
                })
            else:
                complex_content.append({
                    'type': 'document',
                    'source': {
                        'type': 'base64',
                        'media_type': mime_type,
                        'data': base64_data
                    }
                })

        else:
            # text message
            if complex_content is None:
                content_list.append(i)
            else:
                complex_content.append({
                    'type': 'text',
                    'text': i
                })
                content_list.append(complex_content)
                complex_content = None

    return content_list


def create_parser():
    parser = argparse.ArgumentParser(
        description='The script to chat with the agent.')
    parser.add_argument(
        '-c', '--config-path', type=str, required=True,
        help='The path to the config file.')
    parser.add_argument(
        '-o', '--output-path', default=None, type=str,
        help='The path to the save the chat history (if given).')
    parser.add_argument(
        '-q', '--query', nargs='*', help='The begining queries.')
    parser.add_argument(
        '--stream-mode', default='messages', type=str,
        help='The stream mode.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    agent = fvg.common.make_object_from_config(config['agent'])
    stream_config = config.get(
        'stream_config', {'configurable': {'thread_id': 'main'}})
    content_list = make_user_message_content_list(args.query)
    while True:
        content = (
            content_list.pop(0)
            if len(content_list) > 0
            else input('human: ')
        )
        if content == 'q':
            break

        interrupts = agent.get_state(stream_config).interrupts
        graph_output = agent.stream(
            (
                {
                    config.get('messages_key', 'messages'): [
                        {'role': 'user', 'content': content}
                    ]
                }
                if len(interrupts) == 0
                else Command(resume=content)
            ),
            stream_config, stream_mode=args.stream_mode)
        if args.stream_mode == 'messages':
            printer = ChuckPrinter()
            for message_chunk, metadata in graph_output:
                printer.update(message_chunk, metadata)

            print()

        elif args.stream_mode == 'updates':
            for step_output in graph_output:
                for node, response in step_output.items():
                    print_message(node, response)

    if args.output_path:
        state = agent.get_state(stream_config)
        output_data = {
            'config_path': args.config_path,
            config.get('messages_key', 'messages'): convert_to_openai_messages(
                state.values[config.get('messages_key', 'messages')],
                ensure_ascii=False)
        }
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
