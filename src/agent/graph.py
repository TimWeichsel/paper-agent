from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from src.agent.agent_state import AgentState
from src.agent.nodes import paper_assistant, paper_organizer, validator, tools
from src.agent.edges import validator_condition

builder = StateGraph(AgentState)

builder.add_node("paper_assistant", paper_assistant)
builder.add_node("paper_organizer", paper_organizer)
builder.add_node("validator", validator)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "paper_assistant")
builder.add_conditional_edges("paper_assistant", tools_condition, {"tools": "tools", END: "paper_organizer"})
builder.add_edge("tools", "paper_assistant")
builder.add_edge("paper_organizer", "validator")
builder.add_conditional_edges("validator", validator_condition, {"paper_organizer": "paper_organizer", END: END})

graph = builder.compile()

def show_graph():
    # Show
    from IPython.display import Image, display
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))



    

