from typing import Optional, Literal
from langgraph.graph import MessagesState


# Define AgentState as a MessagesState with add_messages reducer (TypedDict)
class AgentState(MessagesState):
    '''Shared graph state for the agents'''
    # UI Input Fields
    query_preference: Optional[str]
    complexity_preferences: Optional[Literal["basic", "medium", "advanced"]]
    learning_mode: Optional[Literal["new", "related"]]

    # Paper Candidates Dict Field
    paper_candidates: Optional[dict] #{Titel: {abstract, author(s), date, citation_count, pdf_link}}

    # Control Fields
    target_concept: Optional[str]

    # Result Fields
    paper_title: Optional[str]
    paper_concept: Optional[str]
    paper_concept_definition: Optional[str]
    concept_summary: Optional[str]

    # Validator Fields
    validator_title_rejection: Optional[bool]
    validator_concept_rejection: Optional[bool]
    validator_concept_definition_rejection: Optional[bool]
    validator_summary_rejection: Optional[bool]
    validator_title_message: Optional[str]
    validator_concept_message: Optional[str]
    validator_concept_definition_message: Optional[str]
    validator_summary_message: Optional[str]
    validator_counter: Optional[int]

    # Saved to DB
    saved_to_db: Optional[bool]
    duplicate_of_concept: Optional[str]