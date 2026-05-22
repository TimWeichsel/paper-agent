temperature=0.3

#Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Optional, Literal, TypedDict

from src.services.knowledge_retriever import load_paper_information
from src.services.arxiv_client import arxiv_paper_search

# Load all environment variables from .env file
load_dotenv()

# Initialize llm with temperature
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
tools = [load_paper_information,arxiv_paper_search]
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage('''You are a machine learning research expert.
    Your goal is to find and explain a paper that extends the user's current ML knowledge.
        1. Use load_paper_information to retrieve the current knowledge base and paper history.
        2. Use arxiv_paper_search to find relevant papers.
        3. Choose the most suitable paper and explain it in detail.
    Focus on papers that build on existing knowledge without repeating it
    Return the following an The Title and the Explanation of the paper''')

# Define AgentState as a TypedDict
class AgentState(MessagesState):
    learning_preferences: Optional[str]
    #paper_information: Optional[str]
    paper_title: Optional[str]
    paper_explanation: Optional[str]
    #paper_base: Optional[str]

def paper_assistant(state: AgentState) -> AgentState:
    result = llm_with_tools.invoke([sys_msg] + state["messages"])
    print(f"LLM Result: {result}")
    return {"messages": [result]}


    

builder = StateGraph(AgentState)

builder.add_node("paper_assistant", paper_assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "paper_assistant")
builder.add_conditional_edges("paper_assistant", tools_condition)
builder.add_edge("tools", "paper_assistant")

graph = builder.compile()

graph.invoke({
    "messages": [HumanMessage("I want to learn about transformers")],
    "learning_preferences": "transformers"
})



    

