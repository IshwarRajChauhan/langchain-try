from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq

## State Schema
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage  ## Human message or AI message
from typing import Annotated  ## labelling
from langgraph.graph.message import add_messages  ## Reducers in LangGraph

### Entire Chatbot With LangGraph
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

import os
from dotenv import load_dotenv
load_dotenv()

#-----TOOL1------------------------------------------------------------------------------------------
api_wrapper_arxiv=ArxivAPIWrapper(top_k_result=2,doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv,description="Query arxiv papers")
print(arxiv.name)

#print(arxiv.invoke("Attention is all you need"))

#-----TOOL2------------------------------------------------------------------------------------------


api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
#print(wiki.name)

#------TOOL3-----------------------------------------------------------------------------------------

tavily = TavilySearch()
#print(tavily.invoke("provide me recent news about india"))




#---Combine all tools-------------------------------------------------------------------------------------
tools = [arxiv,wiki,tavily]



#--LLM---------------------------------------------------------------------------------------

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(tools=tools)


print(llm_with_tools.invoke("when was abhinav bindra born"))

#--Langgraph---------------------------------------------------------------------------------------


class State(TypedDict):
    messages: Annotated[list[AnyMessage],add_messages]




## Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}


#Build Graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))



## Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
)

# builder.add_edge("tools",END)      #for 
builder.add_edge("tools","tool_calling_llm")

graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())








