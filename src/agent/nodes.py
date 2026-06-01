temperature=0.3

#Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from utils.file_utils import load_file, save_file, append_to_file
from src.agent.prompts import create_paper_assistent_sys_msg, create_paper_organizer_sys_msg, create_validator_sys_msg
from langchain_core.messages import HumanMessage, ToolMessage
from src.agent.agent_state import AgentState

from src.services.knowledge_retriever import load_paper_information
from src.services.arxiv_client import arxiv_paper_search
from src.services.serpapi_client import serpapi_paper_search

# Load all environment variables from .env file
load_dotenv()

# Initialize llm with temperature
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
#llm = ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
#llm = ChatMistralAI(model="mistral-large-latest", temperature=temperature)
#llm = ChatOllama(model="mistral", temperature=temperature)

tools = [load_paper_information, arxiv_paper_search, serpapi_paper_search]
llm_with_tools = llm.bind_tools(tools)

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
    
    sys_msg = create_paper_assistent_sys_msg(query_preference, complexity_preference, concept_base, knowledge_base, tool_call_count)

    llm_to_use = llm if tool_call_count >= 3 else llm_with_tools
    print("STARTinvoke")
    print(sys_msg)
    print("END invoke")
    result = llm_to_use.invoke([sys_msg] + last_messages)
    print("START result")
    print(result)
    print("END result")
    return {"messages": [result], "concept_base": concept_base, "validator_title_message": "", "validator_concept_message": "", "validator_summary_message": "", "validator_title_rejection": True, "validator_concept_rejection": True, "validator_summary_rejection": True, "validator_counter": 0}

def paper_organizer(state: AgentState) -> AgentState:
    paper_titel_msg, paper_concept_msg, paper_summary_msg = create_paper_organizer_sys_msg(
        state.get("validator_title_message", ""),
        state.get("validator_concept_message", ""),
        state.get("validator_summary_message", "")
    )

    validator_title_rejection = state.get("validator_title_rejection")
    validator_concept_rejection = state.get("validator_concept_rejection")
    validator_summary_rejection = state.get("validator_summary_rejection")

    if validator_title_rejection:
        paper_title = llm.invoke([paper_titel_msg] + [HumanMessage(content=state["messages"][-1].content)]).content
        print("START title")
        print(paper_title)
        print("END title")
    else: 
        paper_title = state.get("paper_title")

    if validator_concept_rejection:
        paper_concept = llm.invoke([paper_concept_msg] + [HumanMessage(content=state["messages"][-1].content)]).content
        print("START concept")
        print(paper_concept)
        print("END concept")
    else:
        paper_concept = state.get("paper_concept")
    
    if validator_summary_rejection:
        paper_summary = llm.invoke([paper_summary_msg] + [HumanMessage(content=state["messages"][-1].content)]).content
        print("START summary")
        print(paper_summary)
        print("END summary")
    else:
        paper_summary = state.get("paper_summary")

    return {"paper_title": paper_title, "paper_summary": paper_summary, "paper_concept": paper_concept}

def validator(state: AgentState) -> AgentState:
    validator_counter = state.get("validator_counter", 0) + 1
    title_validator_msg, concept_validator_msg, summary_validator_msg = create_validator_sys_msg(state.get("paper_title"), state.get("paper_summary"), state.get("paper_concept"), state["messages"][-1].content)

    
    title_validator_result = llm.invoke([title_validator_msg])
    print("START title_validator_result")
    print(title_validator_result)
    print("END title_validator_result")

    concept_validator_result = llm.invoke([concept_validator_msg])
    print("START concept_validator_result")
    print(concept_validator_result)
    print("END concept_validator_result")

    summary_validator_result = llm.invoke([summary_validator_msg])
    print("START summary_validator_result")
    print(summary_validator_result)
    print("END summary_validator_result")

    title_not_approved = "approve" not in title_validator_result.content.lower()
    concept_not_approved = "approve" not in concept_validator_result.content.lower()
    summary_not_approved = "approve" not in summary_validator_result.content.lower()
    title_rejection_msg = title_validator_result.content if title_not_approved else ""
    concept_rejection_msg = concept_validator_result.content if concept_not_approved else ""
    summary_rejection_msg = summary_validator_result.content if summary_not_approved else ""

    if not title_not_approved and not concept_not_approved and not summary_not_approved:
        append_to_file("src/data/paper_list.txt", "\n\n" + state.get("paper_title"))
        append_to_file("src/data/concept_base.txt", "\n" + state.get("paper_concept") + ": " + state.get("paper_summary"))

    if validator_counter < 3:
        return {"validator_title_rejection": title_not_approved, "validator_concept_rejection": concept_not_approved, "validator_summary_rejection": summary_not_approved, "validator_title_message": title_rejection_msg, "validator_concept_message": concept_rejection_msg, "validator_summary_message": summary_rejection_msg, "validator_counter": validator_counter}
    
    else:
        return {"validator_title_rejection": False, "validator_concept_rejection": False, "validator_summary_rejection": False, "validator_title_message": title_rejection_msg, "validator_concept_message": concept_rejection_msg, "validator_summary_message": summary_rejection_msg, "validator_counter": validator_counter}

