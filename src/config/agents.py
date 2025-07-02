# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Literal
from collections import defaultdict


# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision"]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = defaultdict(lambda: "basic")
AGENT_LLM_MAP["coordinator"] = "basic"
AGENT_LLM_MAP["planner"] = "basic"
AGENT_LLM_MAP["researcher"] = "basic"
AGENT_LLM_MAP["coder"] = "basic"
AGENT_LLM_MAP["reporter"] = "basic"
AGENT_LLM_MAP["podcast_script_writer"] = "basic"
AGENT_LLM_MAP["ppt_composer"] = "basic"
AGENT_LLM_MAP["prose_writer"] = "basic"
AGENT_LLM_MAP["vqa"] = "vision"