from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import tools_condition
import os
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_tool_graph():
    ## Graph With tool Call
    @tool
    def add(a: float, b: float):
        """Add two number"""
        return a + b

    tools = [add]
    tool_node = ToolNode(tools)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    llm_with_tool = llm.bind_tools(tools)


    def call_with_model(state: State):
        return {"messages": [llm_with_tool.invoke(state["messages"])]}


    ## Node definition
    def call_llm_model(state: State):
        return {"messages": [llm_with_tool.invoke(state["messages"])]}

    ## Graph
    builder = StateGraph(State)

    builder.add_node("tool_calling_llm", call_llm_model)
    builder.add_node("tools", ToolNode(tools))

    ## Add Edges
    builder.add_edge(START, "tool_calling_llm")

    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition
    )

    ## compile the graph
    graph = builder.compile()
    return graph


tool_agent = make_tool_graph()