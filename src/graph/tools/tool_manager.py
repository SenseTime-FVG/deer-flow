from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode


class ToolManager:
    """工具管理器 - 统一管理所有工具"""

    def __init__(self, tools: list, node_tools: dict):
        self.tools = {i.name: i for i in tools}
        self.node_tools = node_tools

    def get_tools_for_node(self, node_name: str) -> list:
        """
        获取节点可用的工具
        Args:
            node_name: 节点名称
        Returns:
            List[Any]: 可用的工具列表
        """
        return [
            self.tools[i] for i in self.node_tools[node_name]
            if i in self.tools
        ]

    def build_tool_nodes_and_edges(self, graph: StateGraph, node_dict: dict):
        tool_nodes = {
            node_name: ToolNode(
                self.get_tools_for_node(node_name),
                name="{}_tools".format(node_name),
                messages_key=node_dict[node_name].messages_key)
            for node_name in self.node_tools.keys()
        }
        for node_name, tool_node in tool_nodes.items():
            graph = graph.add_node(tool_node.name, tool_node)\
                .add_edge(tool_node.name, node_name)

        return graph
