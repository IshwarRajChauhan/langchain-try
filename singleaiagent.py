from nt import system
import os
from typing import TypedDict, Annotated, List, Literal
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END, MessagesState,START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()




#State
class AgentState(MessagesState):
    next_agent: str #which agent should go next


@tool
def search_web(query:str) -> str:
    """ Search the web for information."""
    search = TavilySearch(max_result=3)
    results = search.invoke(query)
    return str(results)


@tool
def write_summary(content: str) -> str:
    """Write a summary of the provided content."""
    # Simple summary generation
    summary = f"Summary of findings:\n\n{content[:500]}..."
    return summary

llm = init_chat_model("groq:openai/gpt-oss-20b")


# Define agent functions (simpler approach)
def researcher_agent(state: AgentState):
    """Researcher agent that searches for information"""

    messages = state["messages"]

    # Add system message for context
    system_msg = SystemMessage(content="You are a research assistant. Use the search_web tool to find information")

    # Call LLM with tools
    researcher_llm = llm.bind_tools([search_web])
    response = researcher_llm.invoke([system_msg] + messages)

    # Return the response and route to writer
    return {
        "messages": [response],
        "next_agent": "writer"
    }



def writer_agent(state:AgentState):
    """Writer agent that vreates summaries """
    messages = state["messages"]

    #Add system message
    system_msg = SystemMessage(content="You are a technical writer. Review the conversation")

    #Simple completion without tools
    response = llm.invoke([system_msg] + messages)

    return {
        "messages":[response],
        "next_agent": "end"
    }


# Tool executor node
def execute_tools(state: AgentState):
    """Execute any pending tool calls"""
    messages = state["messages"]
    last_message = messages[-1]

    # Check if there are tool calls to execute
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Create tool node and execute
        tool_node = ToolNode([search_web, write_summary])
        response = tool_node.invoke(state)
        return response

    # No tools to execute
    return state





def create_minimal_multi_agent():
    """Minimal multi-agent system using just LLM calls"""

    def researcher_node(state: MessagesState):
        """Simple researcher using just LLM"""
        messages = state["messages"]

        # Create prompt for researcher
        prompt = [
            SystemMessage(content="You are a researcher. Analyze the user's question and provide detailed"),
            *messages
        ]

        # Get response
        response = llm.invoke(prompt)
        return {"messages":[response]}

    def writer_node(state:MessagesState):
        """Simple writer using just LLM"""
        messages = state["messages"]

        # Create prompt for writer
        prompt = [
            SystemMessage(content="You are a writer. Based on the research provided, create a concise"),
            *messages
        ]

        response = llm.invoke(prompt)
        return {"messages": [response]}

    #Build graph
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)

    # Define flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    final_workflow = workflow.compile()
    print(final_workflow.get_graph().draw_mermaid())

    final_workflow.get_graph().draw_mermaid_png(
        output_file_path="single_agent_graph.png"
    )

    return final_workflow

    
if __name__ == "__main__":
    final_workflow = create_minimal_multi_agent()

    result = final_workflow.invoke({
        "messages": [
            HumanMessage(content="Research about the usecase of agentic AI in business")
        ]
    })

    print(result["messages"][-1].content)