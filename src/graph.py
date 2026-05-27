temperature=0.3

#Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Optional, Literal, TypedDict

from src.services.knowledge_retriever import load_paper_information
from src.services.arxiv_client import arxiv_paper_search

from utils.file_utils import load_file, save_file, append_to_file


# Load all environment variables from .env file
load_dotenv()

# Initialize llm with temperature
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
#llm = ChatMistralAI(model="mistral-large-latest", temperature=temperature)
#llm = ChatOllama(model="mistral", temperature=temperature)

tools = [load_paper_information,arxiv_paper_search]
llm_with_tools = llm.bind_tools(tools)

# System message
def create_sys_msg(query_preference: str, complexity_preference: str = "medium", paper_base: str = "No papers analyzed yet", knowledge_base: str = "No knowledge base found", tool_call_count: int = 0) -> SystemMessage:
    return SystemMessage(f'''You are a research expert. The user wants to learn the following: {knowledge_base}.
    This is the current base of he users known papers: {paper_base}
    Your goal is to find and explain a paper that extends the user's current knowledge or that matches their interests (if they specify a low complex paper than try to find the standard paper in this topic, use your knowledge what the name of the standard paper is).
        1. Use arxiv_paper_search to find relevant papers (consider preferences, complexity, avoid niche papers).
        2. Choose the most suitable paper and explain it in detail.
    The user has the following complexity preference: "{complexity_preference}".
    If this string is not empty, the user has the following query preference: "{query_preference}", focus on this query preference and find papers that build on the specified concepts. Do not direcly use the query preference as your search term but rather adapt find a search query that matches both the query preference and the complexity preference.
    Else, find good papers that match the complexity preference and are relevant to the users current knowledge base.
    You can use up to 3 tool calls to find a good paper, current count is {tool_call_count}.
    Return only the following: The Title and the Explanation of the paper''')

paper_titel_msg = SystemMessage('''You are a research expert.
    Your goal is to extract the title of the given paper information. Don't return anyhing but just the title.''')


paper_base_update_msg = SystemMessage(                  
    '''Read the current knowledge base and expand it with the knowledge of the new paper. Give a full overview of all papers by consiering the current paper informaion and the base. Do not force to find connection but rather summarze the knowledge in a structured way. I don't need to have the information of when which paper was analyzed but rather a structured overview. Here is the base:''')

paper_base_update_msg2 = SystemMessage(                  
    '''And here is the new paper information:''')



# Define AgentState as a TypedDict
class AgentState(MessagesState):
    query_preference: Optional[str]
    complexity_preferences: Optional[str]
    paper_title: Optional[str]
    paper_explanation: Optional[str]
    paper_base: Optional[str]
    paper_assistent_tool_calls: Optional[int]

def paper_assistant(state: AgentState) -> AgentState:
    state["paper_assistent_tool_calls"] = state["paper_assistent_tool_calls"] if state["paper_assistent_tool_calls"] is not None else 0
    messages = state["messages"] or [HumanMessage(content="Find me a paper")]
    last_human_message_idx = next((len(messages)-1-message_idx for message_idx, message in enumerate(reversed(messages)) if isinstance(message, HumanMessage)), None) 
    last_messages = messages[last_human_message_idx:] if last_human_message_idx is not None else [HumanMessage(content="Find me a paper")]
    tool_call_count = sum(1 for m in last_messages if isinstance(m, ToolMessage))
    state["paper_assistent_tool_calls"] = tool_call_count

    query_preference = state.get("query_preference") or ""
    complexity_preference = state.get("complexity_preferences") or "medium"
    knowledge_base = load_file("src/data/knowledge_base.txt", "No knowledge base found")
    paper_base = load_file("src/data/paper_base.txt", "No papers analyzed yet")
    
    sys_msg = create_sys_msg(query_preference, complexity_preference, paper_base, knowledge_base, tool_call_count)
    
    llm_to_use = llm if tool_call_count >= 3 else llm_with_tools
    result = llm_to_use.invoke([sys_msg] + last_messages)
    return {"messages": [result], "paper_base": paper_base}

def paper_organizer(state: AgentState) -> AgentState:
    paper_title = llm.invoke([paper_titel_msg] + [HumanMessage(content=state["messages"][-1].content)])
    append_to_file("src/data/paper_list.txt", paper_title.content)
    old_paper_base  = HumanMessage(content=state["paper_base"])
    new_paper_base = llm.invoke([paper_base_update_msg, old_paper_base, paper_base_update_msg2]+ [HumanMessage(content=state["messages"][-1].content)])
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



    

