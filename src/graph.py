temperature=0.3

#Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Optional, Literal, TypedDict

from src.services.knowledge_retriever import load_paper_information
from src.services.arxiv_client import arxiv_paper_search

from utils.file_utils import load_file, save_file, append_to_file


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

paper_titel_msg = SystemMessage('''You are a machine learning research expert.
    Your goal is to extract the title of the given paper information. Don't return anyhing but just the title.''')


paper_base_update_msg = SystemMessage(                  
    '''Read the current knowledge base and expand it with the knowledge of the new paper. Give a full overview of all papers by consiering the current paper informaion and the base. Here is the base:''')

paper_base_update_msg2 = SystemMessage(                  
    '''And here is the new paper information:''')



# Define AgentState as a TypedDict
class AgentState(MessagesState):
    learning_preferences: Optional[str]
    paper_title: Optional[str]
    paper_explanation: Optional[str]
    paper_base: Optional[str]

def paper_assistant(state: AgentState) -> AgentState:
    result = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [result]}

def paper_organizer(state: AgentState) -> AgentState:
    paper_title = llm.invoke([paper_titel_msg] + state["messages"])
    append_to_file("src/data/paper_list.txt", paper_title.content)
    old_paper_base  = HumanMessage(content=load_file("src/data/paper_base.txt", "No papers analyzed yet"))# Read paper_base.txt
    new_paper_base = llm.invoke([paper_base_update_msg], old_paper_base, paper_base_update_msg2 + state["messages"])
    save_file("src/data/paper_base.txt", new_paper_base.content)
    return {"paper_title": paper_title.content, "paper_base": new_paper_base.content    }

builder = StateGraph(AgentState)

builder.add_node("paper_assistant", paper_assistant)
builder.add_node("paper_organizer", paper_organizer)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "paper_assistant")
builder.add_conditional_edges("paper_assistant", tools_condition, {"tools": "tools", END: "paper_organizer"})
builder.add_edge("tools", "paper_assistant")
builder.add_edge("paper_organizer", END)


graph = builder.compile()



    

