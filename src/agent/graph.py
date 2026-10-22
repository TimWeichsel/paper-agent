from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from src.agent.agent_state import AgentState
from src.agent.nodes import paper_researcher, paper_refiner, validator, tools
from src.agent.edges import validator_condition

#Script that builds the graph for the agents, defining the nodes and edges. At the end it compiles the graph and assigns it to the variable 'graph' (imported in the frontend)

builder = StateGraph(AgentState)

builder.add_node("paper_researcher", paper_researcher)
builder.add_node("paper_refiner", paper_refiner)
builder.add_node("validator", validator)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "paper_researcher")
builder.add_conditional_edges("paper_researcher", tools_condition, {"tools": "tools", END: "validator"})
builder.add_edge("tools", "paper_researcher")
builder.add_edge("paper_refiner", "validator")
builder.add_conditional_edges("validator", validator_condition, {"paper_refiner": "paper_refiner", END: END})

graph = builder.compile()