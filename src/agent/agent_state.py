from typing import Optional, Literal, TypedDict
from langgraph.graph import MessagesState


# Define AgentState as a TypedDict
class AgentState(MessagesState):
    query_preference: Optional[str]
    complexity_preferences: Optional[str]

    paper_title: Optional[str]
    paper_concept: Optional[str]
    paper_summary: Optional[str]

    concept_base: Optional[str]

    paper_assistent_tool_calls: Optional[int]
    
    validator_title_rejection: Optional[bool]
    validator_concept_rejection: Optional[bool]
    validator_summary_rejection: Optional[bool]
    validator_title_message: Optional[str]
    validator_concept_message: Optional[str]
    validator_summary_message: Optional[str]
    validator_counter: Optional[int]