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
def create_sys_msg(query_preference: str, complexity_preference: str = "medium", paper_base: str = "No papers analyzed yet", knowledge_base: str = "No knowledge base found", tool_call_count: int = 0) -> SystemMessage:
    if tool_call_count >= 3:
        return SystemMessage(f'''You are a research expert. The user wants to learn the following: {knowledge_base}.
                             You can see the messages. Choose one paper that matches the best and return only the following: The Title and the Explanation of the paper''')
                             
    return SystemMessage(f'''You are a concept learning expert. The user's existing knowledge base: {knowledge_base}.
    Papers the user has already studied: {paper_base}
    The user wants to deeply understand ONE specific concept — not just read a paper summary, but truly grasp the concept and know how to apply it.

    Your task:
        1. From the query preference or knowledge base, identify ONE concept the user should learn (if a specific concept is mentioned in the query preference, use that; otherwise, find a concept that naturally extends their knowledge or fulfills their general query preference).
        2. Use your own knowledge to determine the single most canonical paper for that concept.
           Examples: "basic ml model" → XGBoost paper or Logistic Regression; "attention" → "Attention Is All You Need"; "gradient boosting" → Friedman 2001.
           Do NOT use the user's raw input as your search term — derive the actual paper concept (or title) first.
        3. Search arxiv_paper_search using the concept/paper title or authors you identified in step 2.
        4. If arxiv_paper_search returns an ERROR, call serpapi_paper_search with the same query — do not stop.
        5. Use the found paper as the anchor for explaining the concept.

    The user has the following complexity preference: "{complexity_preference}".
    If this string is not empty, the user has the following concept/query preference: "{query_preference}". Map this to the foundational or most widely cited paper in that area — think like a professor assigning the must-read paper for that topic.
    Else, identify a concept that naturally extends the user's current knowledge base.

    Return ONLY the following structure:
    **Concept:** <Name of the concept>
    **Paper:** <Title of the paper> — <Authors, Year>
    **What it is:** <2-3 sentences: precise definition of the concept. Cite how the paper itself introduces or defines it.>
    **How it works:** <4-5 sentences: the core mechanism, building from first principles. Reference the key section, figure, or result from the paper that best illustrates this.>
    **Key insight from the paper:** <1-2 sentences: the single most important finding or contribution the paper makes about this concept — what makes this paper the go-to source.>
    **How to apply it:** <3-4 sentences: concrete, practical steps for using this concept. Ground each step in what the paper demonstrates or proves is effective.>
    **Connection to your knowledge:** <1-2 sentences: how this concept and paper relate to what the user already knows>''')

paper_titel_msg = SystemMessage('''You are a research expert.
    Your goal is to extract the title of the given paper information. Don't return anyhing but just the title.''')

paper_concept_msg = SystemMessage('''Extract the analyzed **Concept** of the given summary''')

paper_base_update_msg = SystemMessage(
    '''Please summarize the following paper in 200 words. This summary will be appended to the paper base. Directly start your answerwith the summary without any introduction.''')

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
    paper_base: Optional[str]
    paper_assistent_tool_calls: Optional[int]

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
    paper_base = load_file("src/data/paper_base.txt", "No papers analyzed yet")
    
    sys_msg = create_sys_msg(query_preference, complexity_preference, paper_base, knowledge_base, tool_call_count)

    llm_to_use = llm if tool_call_count >= 3 else llm_with_tools
    print("STARTinvoke")
    print(sys_msg)
    print("END invoke")
    result = llm_to_use.invoke([sys_msg] + last_messages)
    print("START result")
    print(result)
    print("END result")
    return {"messages": [result], "paper_base": paper_base}

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
    
    paper_summary = llm.invoke([paper_base_update_msg] + [HumanMessage(content=state["messages"][-1].content)])
    print("START summary")
    print(paper_summary)
    print("END summary")
    append_to_file("src/data/paper_base.txt", "\n" + paper_concept.content + ": " + paper_summary.content)
    return {"paper_title": paper_title.content, "paper_base": paper_summary.content}

builder = StateGraph(AgentState)

builder.add_node("paper_assistant", paper_assistant)
builder.add_node("paper_organizer", paper_organizer)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "paper_assistant")
builder.add_conditional_edges("paper_assistant", tools_condition, {"tools": "tools", END: "paper_organizer"})
builder.add_edge("tools", "paper_assistant")
builder.add_edge("paper_organizer", END)


graph = builder.compile()



    

