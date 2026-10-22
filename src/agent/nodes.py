from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.prompts import create_paper_researcher_sys_msg, create_paper_refiner_sys_msgs, create_validator_sys_msg
from src.agent.agent_state import AgentState
from src.services.semantic_scholar_client import semantic_scholar_paper_search
from src.services.db_client import add_paper_with_concepts_to_db, get_all_concept_names_from_db, get_research_goal_from_db, get_most_similar_concept

from pydantic import BaseModel, Field
import logging
logger = logging.getLogger(__name__)

MAX_COSINE_DUPLICATE_DISTANCE = 0.1
MAX_TOOL_CALLS = 3
MAX_ATTEMPTED_TOOL_CALLS = 5
MAX_VALIDATOR_LOOPS = 3
NO_REJECTION_DICT = {"validator_title_rejection": False, "validator_concept_rejection": False, "validator_concept_definition_rejection": False, "validator_summary_rejection": False}


# Load all environment variables from .env file
load_dotenv()

class PaperSelection(BaseModel):
    '''Structured output class for paper selection agent'''
    title: str
    concept_name: str
    concept_definition: str
    concept_summary: str

class ResultValidation(BaseModel):
    '''Structured output class for result validation agent'''
    approved: bool = Field(description="Set to true if the result satisfies all objectives, otherwise set to false.")
    reason: str = Field("", description="If approved is false, provide a brief explanation of what is wrong with the result.")


# Initialize LLMs with fixed temperature
TEMPERATURE=0.3
LLM = ChatGroq(model="openai/gpt-oss-120b", temperature=TEMPERATURE, reasoning_effort="low", timeout=60, max_retries = 1)
LLM2 = ChatGroq(model="openai/gpt-oss-20b", temperature=TEMPERATURE, reasoning_effort="low", timeout=60, max_retries = 2)
LLM_BACKUP = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=TEMPERATURE)
LLM_VALIDATOR = LLM2.with_structured_output(ResultValidation).with_fallbacks([LLM_BACKUP.with_structured_output(ResultValidation)])
LLM_REFINER = LLM2.with_fallbacks([LLM_BACKUP])
tools = [semantic_scholar_paper_search]

def paper_researcher(state: AgentState) -> AgentState:
    '''
    Finds a paper for a given concept, complexity, preference, learning mode and previous learned concepts.
    By utilizing the research tools (semantic scholar), the researcher will formulate queries and select a paper that fits the user's needs.
    The researcher has multiple chances to find a good paper and will return a structured output.
    If no paper is found (tool is not working) it might hallucinate. This is why a validator is later used.

    Parameters:
    - state (AgentState): The current state of the agent.

    Returns:
    - state (AgentState): The updated state of the agent with the paper selection.
    '''
    query_preference = state.get("query_preference") or ""
    default_msg = f"I want to learn about {query_preference}" if query_preference else "Find me a new concept to learn"
    messages = state["messages"] or [HumanMessage(content=default_msg)]
    last_human_message_idx = next((len(messages)-1-message_idx for message_idx, message in enumerate(reversed(messages)) if isinstance(message, HumanMessage)), None) # Extract the last human message and all messages from that point on, to run Agent for this call (Previous messages are not relevant for learning a new concept)
    session_messages = messages[last_human_message_idx:] if last_human_message_idx is not None else messages

    logger.debug(f"session_messages: {len(session_messages)}")

    # Update the current tool count of the agent
    tool_call_count = sum(1 for m in session_messages if _is_successful_tool_call(m)) # Only successful tool calls are counted, errors are ignored
    attempted_tool_call_count = sum(1 for m in session_messages if isinstance(m, ToolMessage)) # Count all tool calls, even if they failed

    # Settings and Information for LLM system message 
    llm_to_use, structured_output_flag = _determine_llm_to_use(tool_call_count=tool_call_count, attempted_tool_call_count=attempted_tool_call_count)
    query_preference, complexity_preference, research_goals, concept_names, learning_mode = _extract_state_info_for_sys_msg(state)

    # Create system message and call researcher LLM with messages since last human message and collects all current paper finds
    paper_candidates = _collect_all_paper_candidates_from_tool_messages(session_messages) # Collecting all paper candidates for the current search session
    target_concept = state.get("target_concept","") # If a target concept is set, it will be used for all searches and paper selection. Otherwise, the researcher can choose a new concept.
    sys_msg = create_paper_researcher_sys_msg(query_preference=query_preference, complexity_preference=complexity_preference, concept_names=concept_names, research_goals=research_goals, learning_mode=learning_mode, structured_llm_flag=structured_output_flag, target_concept=target_concept)
    llm_input = [sys_msg] + session_messages if not structured_output_flag else [sys_msg, _collect_all_search_results_from_tool_messages_to_human_message(messages=session_messages, user_msg=default_msg)]
    researcher_response = llm_to_use.invoke(llm_input) # Either a tool call for searching a paper, text (answering no more tool calls are necessary) or a structured paper selection (with the controlled input)
 
    logger.debug(f"researcher sys_msg: {sys_msg}")
    logger.debug(f"researcher_response: {researcher_response}")

    # Checks if the researcher LLM is done and either returns the structured output dict or saves the last messages and paper candidates to the state
    if isinstance(researcher_response, PaperSelection) or not researcher_response.tool_calls: #llm is done
        if not isinstance(researcher_response, PaperSelection): #llm delivered not a structured output (another last call is needed)
            llm_structured,structured_output_flag = _determine_llm_to_use(float('inf'), float('inf')) # Use LLM forced to structured output
            sys_msg = create_paper_researcher_sys_msg(query_preference=query_preference, complexity_preference=complexity_preference, concept_names=concept_names, research_goals=research_goals, learning_mode=learning_mode, structured_llm_flag=structured_output_flag, target_concept=target_concept)
            researcher_response = llm_structured.invoke([sys_msg, _collect_all_search_results_from_tool_messages_to_human_message(messages=session_messages, user_msg=default_msg)]) #Not showing any tool call in prompt, otherwise LLM might try to call a tool

        messages = [] if state["messages"] else [HumanMessage(content=default_msg)] #Makes sure that the messages are never empty
        return {
            **NO_REJECTION_DICT,
            "messages": messages,
            "paper_title": researcher_response.title,
            "paper_concept": researcher_response.concept_name,
            "paper_concept_definition": researcher_response.concept_definition,
            "concept_summary": researcher_response.concept_summary,
            "paper_candidates": paper_candidates,
            "validator_title_message": "",
            "validator_concept_message": "",
            "validator_concept_definition_message": "",
            "validator_summary_message": "",
            "validator_counter": 0
        }

    else: #llm is not done
        if not target_concept: #If no target concept is set, it will be set
            target_concept = researcher_response.tool_calls[0]["args"].get("concept", "")
        new_messages = [researcher_response] if state["messages"] else [HumanMessage(content=default_msg), researcher_response] #First run, include the default message, so that messages start with HumanMessage (including the user's query preference)
        return {
            "messages": new_messages,
            "paper_candidates": paper_candidates,
            "target_concept": target_concept
        }
def _determine_llm_to_use(tool_call_count:int, attempted_tool_call_count:int, tools: list = tools, max_tool_calls = MAX_TOOL_CALLS, max_attempted_tool_calls = MAX_ATTEMPTED_TOOL_CALLS) -> tuple:
    '''Determines which llm type to use based on the number of tool calls.
    - llm_mandatory_tools = llm that must use research tools (first call: attempted_tool_call_count = 0)
    - llm_with_tools = standard llm using research tools (tool counts < max_tool_calls)
    - llm_structured_output = llm that must return structured output and no tool calls possible (tool counts >= max_tool_calls or attempted_tool_call_count >= MAX_ATTEMPTED_TOOL_CALLS)
    Parameters:
    - tool_call_count (int): The number of times the tool has been called successfully.
    - attempted_tool_call_count (int): The number of times the tool has been attempted to be called.
    - tools (list): The list of available tools.
    - max_tool_calls (int): The maximum number of tool calls allowed before switching to structured output.
    - max_attempted_tool_calls (int): The maximum number of attempted tool calls allowed before switching to structured output.
    Returns:
    - tuple: A tuple containing the appropriate LLM and a boolean indicating if structured output is required.

    '''
    llm_with_tools = LLM.bind_tools(tools).with_fallbacks([LLM_BACKUP.bind_tools(tools)])
    llm_mandatory_tools = LLM.bind_tools(tools, tool_choice="required").with_fallbacks([LLM_BACKUP.bind_tools(tools, tool_choice="any")])
    llm_structured_output = LLM.with_structured_output(PaperSelection, method="json_schema").with_fallbacks([LLM_BACKUP.with_structured_output(PaperSelection)])

    if tool_call_count >= max_tool_calls or attempted_tool_call_count >= max_attempted_tool_calls: #if the tool has been called max_tool_calls times or max_attempted_tool_calls times, use structured output llm
        return llm_structured_output, True
    elif attempted_tool_call_count==0: #first llm needs to call a tool
        return llm_mandatory_tools, False
    else:
        return llm_with_tools, False #llm can call a tool again, but can also stop
    
def _collect_all_paper_candidates_from_tool_messages(messages: list) -> dict:
    '''Collects all paper candidates from tool messages to a dict.'''
    paper_candidates = {}
    for message in messages:
        if _is_successful_tool_call(message):
            paper_candidates.update(message.artifact)
    return paper_candidates
    
def _extract_state_info_for_sys_msg(state: AgentState) -> tuple[str, str, str, str, str]:
    '''
    Loads all relevant user information from state and DB

    Parameters:
    - state (AgentState): The current state of the agent.

    Returns:
    - query preference: The user's query preference.
    - complexity preference: The user's complexity preference.
    - research goals: The user's research goals.
    - concept names: The user's learned concepts from DB (newline separated).
    - learning mode: The user's learning mode.
    '''
    query_preference = state.get("query_preference") or ""
    complexity_preference = state.get("complexity_preferences") or "basic"
    research_goals = get_research_goal_from_db()
    concept_names = "\n".join(get_all_concept_names_from_db())
    learning_mode = state.get("learning_mode") or "new"
    return query_preference, complexity_preference, research_goals, concept_names, learning_mode

def _is_successful_tool_call(message) -> bool:
    '''Checks a message whether it is a successful tool call (ToolMessage)'''
    if not isinstance(message, ToolMessage):
        return False
    if isinstance(message.artifact, dict) and bool(message.artifact): #non-empty dict check
        return True
    return False

def _collect_all_search_results_from_tool_messages_to_human_message(messages: list, user_msg: str) -> HumanMessage:
    '''Creates a new human message with all paper candidates and an user prompt.'''
    search_results = "\n".join(message.content for message in messages if _is_successful_tool_call(message))
    return HumanMessage(content=f"{user_msg}\nSearch results:\n{search_results}")


def validator(state: AgentState) -> AgentState:
    '''
    Checks the paper title, concept name, concept definition and concept summary given by the researcher agent.
    If all are approved, the paper and its concept is saved to the database.
    Else the validator agents returns the problem given to refiner to adapt the information.

    Parameters:
    - state (AgentState): The current state of the agent.
    
    Returns:
    - state (AgentState): The updated state of the agent with the paper selection.
    '''
    validator_counter = state.get("validator_counter", 0) + 1
    max_validator_loops_reached = validator_counter >= MAX_VALIDATOR_LOOPS
    concept_validator_msg, concept_definition_validator_msg, summary_validator_msg = create_validator_sys_msg(paper_title=state.get("paper_title"), concept=state.get("paper_concept"), concept_definition=state.get("paper_concept_definition"), concept_summary=state.get("concept_summary"), )

    # Check directly whether the paper title is in the paper candidates (no llm needed for this)
    title_lookup = {title.lower(): title for title in state.get("paper_candidates", {})} #current lowercase titles
    matched_title = title_lookup.get((state.get("paper_title") or "").lower(), None) #match lowercase title
    title_approved = matched_title is not None
    title_rejection_msg = f"The title '{state.get('paper_title')}' was not found in the search results. Choose one exactly from the list." if not title_approved else ""

    approved_flags = {"concept_validator": None, "concept_definition_validator": None, "summary_validator": None}
    rejection_msgs = {"concept_validator": "", "concept_definition_validator": "", "summary_validator": ""}
    # Check the concept name, definition and summary with the llm_validator
    for validator_name, msg in {"concept_validator": concept_validator_msg, "concept_definition_validator": concept_definition_validator_msg, "summary_validator": summary_validator_msg}.items():
        result = LLM_VALIDATOR.invoke([msg])
        approved_flags[validator_name] = result.approved
        rejection_msgs[validator_name] = result.reason if not result.approved else ""
        logger.debug(f"validator {validator_name}: approved={result.approved}, reason={result.reason}")

    if title_approved and approved_flags["concept_validator"] and approved_flags["concept_definition_validator"] and approved_flags["summary_validator"]:
        paper_title = matched_title
        duplicate_concept = _find_duplicate_concept(state)
        if duplicate_concept:
            return {**NO_REJECTION_DICT, "validator_counter": validator_counter, "saved_to_db": False, "duplicate_of_concept": duplicate_concept}
        paper_candidates = state.get("paper_candidates", {})
        paper_metadata = paper_candidates.get(paper_title, None)
        authors = paper_metadata.get("author(s)", [])
        publication_date = paper_metadata.get("date", "")
        abstract = paper_metadata.get("abstract", "")
        citation_count = paper_metadata.get("citation_count", 0)
        pdf_link = paper_metadata.get("pdf_link", "")
        
        
        saved_to_db = add_paper_with_concepts_to_db(paper_title = paper_title, paper_abstract = abstract, paper_pdf_link = pdf_link, paper_publication_date = publication_date, paper_citation_count = citation_count, concept = state.get("paper_concept"), concept_summary = state.get("concept_summary"), concept_definition = state.get("paper_concept_definition"), authors = authors)
        # Indicate that the paper and concept have been saved to the database and reset the validator information
        return {**NO_REJECTION_DICT, "validator_counter": validator_counter, "saved_to_db": saved_to_db}
    
    return {
        "validator_title_rejection": not title_approved and not max_validator_loops_reached, 
        "validator_concept_rejection": not approved_flags["concept_validator"] and not max_validator_loops_reached, 
        "validator_concept_definition_rejection": not approved_flags["concept_definition_validator"] and not max_validator_loops_reached, 
        "validator_summary_rejection": not approved_flags["summary_validator"] and not max_validator_loops_reached, 
        "validator_title_message": title_rejection_msg, 
        "validator_concept_message": rejection_msgs["concept_validator"], 
        "validator_concept_definition_message": rejection_msgs["concept_definition_validator"], 
        "validator_summary_message": rejection_msgs["summary_validator"], 
        "validator_counter": validator_counter, 
        "saved_to_db": False
        }

def _find_duplicate_concept(state: AgentState, max_distance:float=MAX_COSINE_DUPLICATE_DISTANCE) -> str | None:
    most_similar = get_most_similar_concept(concept=state.get("paper_concept"), concept_definition=state.get("paper_concept_definition"))
    if most_similar is None:
        return None
    most_similar_concept, cosine_dist = most_similar
    duplicate_concept_found = most_similar_concept is not None and cosine_dist <= max_distance
    logger.info(f"Most similar concept is a duplicate: {duplicate_concept_found}, most_similar_concept: {most_similar_concept}, cosine_dist: {cosine_dist}")
    return most_similar_concept if duplicate_concept_found else None

def paper_refiner(state: AgentState) -> AgentState:
    '''
    Receives the rejection messages for the paper title, concept name, concept definition and/or concept summary from the validator agent.
    The refiner agent will adapt the information and return a new version of the paper title, concept name, concept definition and/or concept summary.

    Parameters:
    - state (AgentState): The current state of the agent.
    
    Returns:
    - state (AgentState): The updated state of the agent with the paper selection.
    '''
    general_error_msg = "Note that the validator noted an error in your last try:"
    validator_title_msg = state.get("validator_title_message", "")
    validator_title_msg = f"{general_error_msg} {validator_title_msg}" if validator_title_msg != "" else ""
    validator_concept_msg = state.get("validator_concept_message", "")
    validator_concept_msg = f"{general_error_msg} {validator_concept_msg}" if validator_concept_msg != "" else ""
    validator_concept_definition_msg = state.get("validator_concept_definition_message", "")
    validator_concept_definition_msg = f"{general_error_msg} {validator_concept_definition_msg}" if validator_concept_definition_msg != "" else ""
    validator_summary_msg = state.get("validator_summary_message", "")
    validator_summary_msg = f"{general_error_msg} {validator_summary_msg}" if validator_summary_msg != "" else ""

    title_sys_msg, paper_concept_msg, paper_concept_definition_msg, concept_summary_msg = create_paper_refiner_sys_msgs(validator_title_msg,validator_concept_msg,validator_concept_definition_msg,validator_summary_msg)

    validator_title_rejection = state.get("validator_title_rejection")
    validator_concept_rejection = state.get("validator_concept_rejection")
    validator_concept_definition_rejection = state.get("validator_concept_definition_rejection")
    validator_summary_rejection = state.get("validator_summary_rejection")

    if validator_title_rejection:
        all_candidate_titles = "\n".join(state.get("paper_candidates", {}).keys())
        title_context = f"All available paper titles:\n{all_candidate_titles}\n The title you extracted was: {state.get('paper_title', 'No title extracted')}"
        paper_title = LLM_REFINER.invoke([title_sys_msg] + [HumanMessage(content=title_context)]).content.strip().strip('"\'')
        logger.debug(f"refined paper_title: {paper_title}")
    else: 
        paper_title = state.get("paper_title")

    if validator_concept_rejection:
        concept_context = f"Paper title: {paper_title}\nCurrent concept: {state.get('paper_concept')}"
        paper_concept = LLM_REFINER.invoke([paper_concept_msg] + [HumanMessage(content=concept_context)]).content.strip().strip('"\'')
        logger.debug(f"refined paper_concept: {paper_concept}")
    else:
        paper_concept = state.get("paper_concept")

    if validator_concept_definition_rejection:
        concept_definition_context = f"Paper title: {paper_title}\nTarget concept: {paper_concept}\nCurrent concept definition: {state.get('paper_concept_definition')}"
        concept_definition = LLM_REFINER.invoke([paper_concept_definition_msg] + [HumanMessage(content=concept_definition_context)]).content.strip()
        logger.debug(f"refined paper_concept_definition: {concept_definition}")
    else:
        concept_definition = state.get("paper_concept_definition")
    
    if validator_summary_rejection:
        abstract = state.get("paper_candidates", {}).get(paper_title, {}).get("abstract", "")
        summary_context = f"Paper title: {paper_title}\nCurrent summary: {state.get('concept_summary')}\nCurrent abstract: {abstract}"
        concept_summary = LLM_REFINER.invoke([concept_summary_msg] + [HumanMessage(content=summary_context)]).content.strip()
        logger.debug(f"refined concept_summary: {concept_summary}")
    else:
        concept_summary = state.get("concept_summary")

    return {"paper_title": paper_title, "concept_summary": concept_summary, "paper_concept_definition": concept_definition, "paper_concept": paper_concept}
