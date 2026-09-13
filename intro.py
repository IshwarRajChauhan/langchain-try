from typing_extensions import TypedDict
import random   
from typing import Literal


from IPython.display import display, Image
from langgraph.graph import StateGraph,START,END





class State(TypedDict):
    graph_info: str
    



def start_play(state: State):
    print("Start play node has been called")
    return {"graph_info":state["graph_info"] + "I am planning to play "}


def cricket(state: State):
    print("Cricket node has been called")
    return {"graph_info":state["graph_info"] + "cricket"}

    
def badminton(state: State):
    print("Badminton node has been called")
    return {"graph_info":state["graph_info"] + "badminton"}
    




def random_play(state:State)-> Literal["cricket", "badminton"]:
    if random.random() > 0.5:
        return "cricket"
    else:
        return "badminton"





##Build Graph
graph = StateGraph(State)


##Add all the nodes to the graph
graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)



##Schedule the flow in the graph
graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play", random_play)
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)


##Compile the graph
graph_builder = graph.compile()


##View
display(Image(graph_builder.get_graph().draw_mermaid_png()))
with open("graph.png", "wb") as f:
    f.write(graph_builder.get_graph().draw_mermaid_png())




print(graph_builder.invoke({"graph_info": "My name is IRC "}))