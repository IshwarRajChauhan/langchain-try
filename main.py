from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import  StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
load_dotenv()



class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]




## Graph With tool Call
@tool
def add(a:float,b:float):
    """Add two number"""
    return a+b

tool = [add]
tool_node = ToolNode(tool)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tool = llm.bind_tools(tool)



def call_with_model(state:State):
    return {"messages": [llm_with_tool.invoke(state["messages"])]}



## Node definition
def call_llm_model(state: State):
    return {"messages": [llm_with_tool.invoke(state["messages"])]}


## Graph
builder = StateGraph(State)

builder.add_node("tool_calling_llm", call_llm_model)
builder.add_node("tools", ToolNode(tool))

## Add Edges
builder.add_edge(START, "tool_calling_llm")

builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)

## compile the graph
graph = builder.compile()

# png = graph.get_graph().draw_mermaid_png()

# with open("graph.png", "wb") as f:
#     f.write(png)

# print("Graph saved as graph.png")


response = graph.invoke({"messages":"tell me about new openai Astra model"})

print(response["messages"][-1].content)
