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
from src.services.serpapi_client import serpapi_paper_search

from utils.file_utils import load_file, save_file, append_to_file


# Load all environment variables from .env file
load_dotenv()

# Initialize llm with temperature
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
#llm = ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
#llm = ChatMistralAI(model="mistral-large-latest", temperature=temperature)
#llm = ChatOllama(model="mistral", temperature=temperature)

tools = [load_paper_information,arxiv_paper_search, serpapi_paper_search]
llm_with_tools = llm.bind_tools(tools)

# System message
def create_sys_msg(query_preference: str, complexity_preference: str = "medium", concept_base: str = "No concepts learned yet", knowledge_base: str = "No knowledge base found", tool_call_count: int = 0) -> SystemMessage:
    if tool_call_count >= 3:
        return SystemMessage(f'''You are a concept learning expert. Concepts the user already knows: {concept_base}. Knowledge base: {knowledge_base}.
    Complexity: "{complexity_preference}". Query: "{query_preference}".

    Using the search results above, return ONLY this structure:
    **Concept:** <concept name>
    **What it is:** <2-3 sentences: plain, precise definition — no examples, no analogies>
    **Connection to your knowledge:** <1 sentence: how this relates to what the user already knows>
    **Paper:** <title> — <authors, year>: <1-2 sentences on where and how this concept is concretely used or introduced in this paper>''')

    return SystemMessage(f'''You are a research assistant.
    User knowledge base: {knowledge_base}
    Concepts already learned: {concept_base}
    Complexity preference: "{complexity_preference}"
    Query: "{query_preference}"

    Based on what the user already knows and has learned, identify the next concept they should learn.
    The concept must fit the complexity preference relative to their current level — if they know basic concepts, pick something one step further; if they know advanced concepts, pick accordingly.
    If a specific query is given, find a concept within that direction that matches their level.
    Then identify the canonical paper for that concept and call arxiv_paper_search with the paper title. On ERROR call serpapi_paper_search.
    Do NOT use the raw user query as search term — derive the actual paper title first.''')

paper_titel_msg = SystemMessage('''You are a research expert.
    Your goal is to extract the title of the given paper information. Don't return anyhing but just the title.''')

paper_concept_msg = SystemMessage('''Extract the analyzed **Concept** of the given summary''')

concept_base_update_msg = SystemMessage(
    '''From the following explanation, write exactly two things — no introduction, start directly:
    1. 1-2 sentences defining the concept precisely in plain terms. No examples, no analogies.
    2. One sentence: "Used in: [paper title] — [what role the concept plays in that paper]."''')

paper_base_update_msg_old = SystemMessage(                  
    '''Read the current knowledge base and expand it with the knowledge of the new paper. Give a full overview of all papers by consiering the current paper informaion and the base. Do not force to find connection but rather summarze the knowledge in a structured way. I don't need to have the information of when which paper was analyzed but rather a structured overview. Here is the base:''')

paper_base_update_msg2_old = SystemMessage(                  
    '''And here is the new paper information:''')



# Define AgentState as a TypedDict
class AgentState(MessagesState):
    query_preference: Optional[str]
    complexity_preferences: Optional[str]
    paper_title: Optional[str]
    paper_explanation: Optional[str]
    concept_base: Optional[str]
    paper_assistent_tool_calls: Optional[int]
    paper_concept: Optional[str]

def paper_assistant(state: AgentState) -> AgentState:
    messages = state["messages"] or [HumanMessage(content="Find me a paper")]
    last_human_message_idx = next((len(messages)-1-message_idx for message_idx, message in enumerate(reversed(messages)) if isinstance(message, HumanMessage)), None) 
    last_messages = messages[last_human_message_idx:] if last_human_message_idx is not None else messages
    print("START last_messages")
    print(last_messages)
    print("END last_messages")

    print("START messages")
    print(messages)
    print("END messages")

    tool_call_count = sum(1 for m in last_messages if isinstance(m, ToolMessage))
    state["paper_assistent_tool_calls"] = tool_call_count

    query_preference = state.get("query_preference") or ""
    complexity_preference = state.get("complexity_preferences") or "medium"
    knowledge_base = load_file("src/data/knowledge_base.txt", "No knowledge base found")
    concept_base = load_file("src/data/concept_base.txt", "No concepts analyzed yet")
    
    sys_msg = create_sys_msg(query_preference, complexity_preference, concept_base, knowledge_base, tool_call_count)

    llm_to_use = llm if tool_call_count >= 3 else llm_with_tools
    print("STARTinvoke")
    print(sys_msg)
    print("END invoke")
    result = llm_to_use.invoke([sys_msg] + last_messages)
    print("START result")
    print(result)
    print("END result")
    return {"messages": [result], "concept_base": concept_base}

def paper_organizer(state: AgentState) -> AgentState:
    paper_title = llm.invoke([paper_titel_msg] + [HumanMessage(content=state["messages"][-1].content)])
    print("START title")
    print(paper_title)
    print("END title")
    append_to_file("src/data/paper_list.txt", "\n\n" + paper_title.content)

    paper_concept = llm.invoke([paper_concept_msg] + [HumanMessage(content=state["messages"][-1].content)])
    print("START concept")
    print(paper_concept)
    print("END concept")
    
    paper_summary = llm.invoke([concept_base_update_msg] + [HumanMessage(content=state["messages"][-1].content)])
    print("START summary")
    print(paper_summary)
    print("END summary")
    append_to_file("src/data/concept_base.txt", "\n" + paper_concept.content + ": " + paper_summary.content)
    return {"paper_title": paper_title.content, "concept_base": paper_summary.content, "paper_concept": paper_concept.content}

def paper_validator(state: AgentState) -> AgentState:

    return state

builder = StateGraph(AgentState)

builder.add_node("paper_assistant", paper_assistant)
builder.add_node("paper_organizer", paper_organizer)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "paper_assistant")
builder.add_conditional_edges("paper_assistant", tools_condition, {"tools": "tools", END: "paper_organizer"})
builder.add_edge("tools", "paper_assistant")
builder.add_edge("paper_organizer", END)


graph = builder.compile()



    

