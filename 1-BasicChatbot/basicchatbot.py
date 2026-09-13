from typing import Annotated
from IPython.display import Image, display
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
load_dotenv()


from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------------------------------------------------------------

memory = MemorySaver()


class State(TypedDict):
    # Messages have the type "list". The "add_messages" function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# -----------------------------------------------------------------------------------


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# -----------------------------------------------------------------------------------



def chatbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}


# -----------------------------------------------------------------------------------


graph_builder.add_node("llmchatbot",chatbot)

graph_builder.add_edge(START,"llmchatbot")
graph_builder.add_edge("llmchatbot",END)


graph = graph_builder.compile()

# -----------------------------------------------------------------------------------


# display(Image(graph.get_graph().draw_mermaid_png()))

# with open("graph.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())

# -----------------------------------------------------------------------------------


response = graph.invoke({"messages":"Hi"})
# print(response["messages"][-1].content)


tool=TavilySearch(max_result=2)
# print(tool.invoke("What is langgraph"))


# -----------------------------------------------------------------------------------



def multiply(a:int,b:int)->int:
    """
    Multiply a and b

    Args:
        a (int): first int
        b (int): second int

    Returns:
        int: output int
    """
    return a*b

tools = [tool,multiply]


llm_with_tools = llm.bind_tools(tools)


def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

# -----------------------------------------------------------------------------------



builder = StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))


builder.add_edge(START, "tool_calling_llm")

builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tool_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition
)


builder.add_edge("tools", "tool_calling_llm")


graph = builder.compile(checkpointer=memory)


# -----------------------------------------------------------------------------------


# display(Image(graph.get_graph().draw_mermaid_png()))

# with open("graph.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())

##-----------------------------------------------------------------------------------
 
# response = graph.invoke({"messages": "Give me the recent ai news and then multiply 5 by 10"})

# for m in response["messages"]:
#     m.pretty_print()





#---check memory issue-----------------------------------------------------------------------------------

# response1 = graph.invoke({"messages": "Hello my name is Ishwar"})
# for m in response1["messages"]:
#     m.pretty_print()


# response2 = graph.invoke({"messages": "What is my name?"})
# for m in response2["messages"]:
#     m.pretty_print()


# ------Memory Fix-----------------------------------------------------------------------------


config = {"configurable": {"thread_id": "1"}}

graph.invoke({"messages": "Hi my name is Ishwar"},config = config)

response = graph.invoke({"messages": "Hi , what is my name?"},config = config)
print(response["messages"][-1].content)


# ------Streaming-----------------------------------------------------------------------------


def superbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}


graph = StateGraph(State)

# node
graph.add_node("SuperBot", superbot)

# Edges
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)
graph_builder = graph.compile(checkpointer=memory)


#-----------------------------------------------------------------------------

# Invocation
# config = {"configurable": {"thread_id": "1"}}
# graph_builder.invoke({"messages": "Hi, My name is Krish And I like cricket"},config)
 

#-with stream----------------------------------------------------------------------------



# Create a thread
config = {"configurable": {"thread_id": "3"}}

# for chunk in graph_builder.stream({"messages": "Hi,My name is Krish And I like cricket"}, config, stream_mode="updates"):      #appends only output
#     print(chunk)


# for chunk in graph_builder.stream({"messages": "Hi,My name is Krish And I like cricket"}, config, stream_mode="values"):      #appends whole conversation
#     print(chunk)

#----with astream-------------------------------------------------------------------------

config = {"configurable": {"thread_id": "5"}}

async def main():
    async for event in graph_builder.astream_events(
        {"messages": ["Hi My name is Ishwar and I like to play cricket"]},
        config,
        version="v2"
    ):
        print(event)

import asyncio
asyncio.run(main())

#-----------------------------------------------------------------------------














