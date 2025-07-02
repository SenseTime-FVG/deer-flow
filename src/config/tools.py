# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import enum
from dotenv import load_dotenv

load_dotenv()


class SearchEngine(enum.Enum):
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    BRAVE_SEARCH = "brave_search"
    ARXIV = "arxiv"
    VOLCANO = "volcano"
    SOGOU = "sogou" 


# Tool configuration
SELECTED_SEARCH_ENGINE = "sogou" # os.getenv("SEARCH_API", SearchEngine.VOLCANO.value)
print(SELECTED_SEARCH_ENGINE)


class RAGProvider(enum.Enum):
    RAGFLOW = "ragflow"


SELECTED_RAG_PROVIDER = os.getenv("RAG_PROVIDER")
