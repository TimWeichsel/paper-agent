from src.agent.agent_state import AgentState
from langgraph.graph import END

def validator_condition(state: AgentState) -> str:
    if state.get("validator_title_rejection") or state.get("validator_concept_rejection") or state.get("validator_summary_rejection"):
        return "paper_organizer"
    return END