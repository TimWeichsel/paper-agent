from src.agent.agent_state import AgentState
from langgraph.graph import END

REJECTION_KEYS = ["validator_title_rejection",
    "validator_concept_definition_rejection",
    "validator_concept_rejection",
    "validator_summary_rejection"
]

def validator_condition(state: AgentState) -> str:
    '''Determines the flow of the graph based on the validation results, it either directs to the paper_refiner node if any validation fails or to the END node otherwise.'''
    if any (state.get(flag) for flag in REJECTION_KEYS):
        return "paper_refiner"
    return END