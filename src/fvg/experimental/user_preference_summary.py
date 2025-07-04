import argparse
import fvg.common
import json
from langchain_core.language_models import LanguageModelLike


default_summary_prompt_template = """
请根据以下聊天历史，总结人类用户 (whose role is user) 偏好，反思在同类问题解决的过程中，是否存在一些做法可避免错误，输出的内容仅包含**精炼的短句，总量不超过 200 字**。
注意：对一类问题而不是对特定问题进行总结。
聊天历史：
{chat_history}
"""

default_accumulative_summary_prompt_template = """
请根据以下聊天历史，更新人类用户 (whose role is user) 偏好，反思在同类问题解决的过程中，是否存在一些做法可避免错误，输出的内容仅包含**精炼的短句，总量不超过 200 字**，兼顾历史偏好。
注意：对一类问题而不是对特定问题进行总结。
聊天历史：
{chat_history}
原有的用户偏好总结：
{user_preference}
"""


def summary(
    llm: LanguageModelLike, chat_history: list,
    prompt_template: str = default_summary_prompt_template
):
    prompt = prompt_template.format_map({
        'chat_history': json.dumps(chat_history, ensure_ascii=False)
    })
    response = llm.invoke([{'role': 'user', 'content': prompt}])
    return response.content


def accumulative_summary(
    llm: LanguageModelLike, chat_history: list, last_summary: str,
    prompt_template: str = default_accumulative_summary_prompt_template
):
    prompt = prompt_template.format_map({
        'chat_history': json.dumps(chat_history, ensure_ascii=False),
        'user_preference': last_summary
    })
    response = llm.invoke([{'role': 'user', 'content': prompt}])
    return response.content


def create_parser():
    parser = argparse.ArgumentParser(
        description='The script to analyse for the user preference.')
    parser.add_argument(
        '-c', '--config-path', type=str, required=True,
        help='The path to the config file.')
    parser.add_argument(
        '-i', '--input-path', type=str, required=True,
        help='The path of the message history file.')
    parser.add_argument(
        '-o', '--output-path', type=str, required=True,
        help='The path to the save the summary of the user preference.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    llm = fvg.common.make_object_from_config(config['llm'])

    with open(args.input_path, 'r', encoding='utf-8') as f:
        messages = (
            eval(config['list_maker'])(json.load(f))
            if 'list_maker' in config else json.load(f)
        )

    chunk_size = config['chunk_size']
    regroupped_messages = [
        [
            k
            for j in range(i, min(i + chunk_size, len(messages)))
            for k in messages[j]
            if k['role'] != 'system'
        ]
        for i in range(0, len(messages), chunk_size)
    ]

    last_summary = None
    for i in regroupped_messages:
        last_summary = (
            summary(llm, i) if last_summary is None
            else accumulative_summary(llm, i, last_summary)
        )
        print(last_summary)
        print('---')

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(last_summary, f, **config['dump_args'])
