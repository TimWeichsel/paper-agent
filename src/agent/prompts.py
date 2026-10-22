from typing import Literal
from langchain_core.messages import SystemMessage

DEFINITION_PROMPT = "concept_definition: 1-2 sentences defining the concept precisely in plain terms. No examples, no analogies."
CONCEPT_NAME_PROMPT = "concept_name: Short, canonical name of a concept in plain text, no sentence, no markdown, no explanation - it may include abbreviations, hyphens, or uppercase letters."
SUMMARY_PROMPT = "4 sentences defining the concept precisely."
EXAMPLE_PROMPT = "A concrete example (2-3 sentences) of how an expert applies this concept in research, including the data/task and why the concept is used."
USE_CASE_PROMPT = "One sentence in the format: \"Used in: [paper title] — [role of concept in paper]\"."

def create_paper_researcher_sys_msg(query_preference: str, complexity_preference: Literal["basic", "medium", "advanced"] = "medium", concept_names: str = "No concepts learned yet", research_goals: str = "No research goals found", learning_mode: Literal["new", "related"] = "new", structured_llm_flag: bool = False, target_concept: str = "") -> SystemMessage:
    '''Researcher prompt: search papers for the (fixed) concept, or fill PaperSelection if structured_llm_flag is True.'''    
    concept_names = concept_names or "None Concept defined yet"
    fixed_concept_msg = f"The chosen concept (FIXED, do NOT change it!) is {target_concept}. Do NOT switch to a different concept. Use this concept for all searches and paper selection." if target_concept else ""
    if structured_llm_flag:
        return SystemMessage(f'''You are a concept learning expert.
            {fixed_concept_msg}
            Based on the search results, fill in the following fields:
            - title: The EXACT paper title as it appears in the search results. Copy it character by character.
            - {CONCEPT_NAME_PROMPT}
            - {DEFINITION_PROMPT}
            - concept_summary: 3 parts — (1) {SUMMARY_PROMPT} (2) {EXAMPLE_PROMPT} (3) {USE_CASE_PROMPT}
            
            Do NOT call any tools. Choose from the search results above.
            ''')

    if learning_mode == "new":
        mode_instruction = "Pick a concept the user has NOT learned yet — it must be completely new and have a different name than any concept listed above."
    else:
        mode_instruction = "Pick a concept in the same field as one of the concepts above, but a genuinely different concept — not a subcategory, variation, or rephrasing of an already-learned concept."

    concept_text = fixed_concept_msg if target_concept else f'''User Query for the next concept: "{query_preference}" (If this string is not empty, then the user wants to learn about this specific concept, which you have to consider and this is a strict requirement!)
    Concepts already learned (do NOT pick these or any variation of them): {concept_names}.
    Identify the next concept the user should learn. It must be a standalone, well-known concept — NOT a broad field or topic cluster. 
    Translate the user query into the established, canonical name of the concept as used in the literature (e.g. "Compression SVD" → "Truncated SVD", "knn" → "K-Nearest Neighbors"). 
    Use this canonical name as concept. \n Instructions:{mode_instruction}'''
    
    return SystemMessage(f'''You are a research assistant.
    User research goals: {research_goals}
    Complexity preference: "{complexity_preference}"

    Your Task:
    {concept_text}
    Find the canonical paper and call semantic_scholar_paper_search.
    If no canonical paper is found, you can also try a well-cited survey or tutorial paper that explains the concept in detail.
    If you have found already a sufficient paper for your concept, you can stop searching (no tool call needed!). You can just answer with DONE!
    If results are irrelevant, try a different paper for the same concept.
    Never repeat a search query you already used: each search must use a different query.
    Do NOT use the raw user query as the search term — derive the actual paper title first.''')

def create_validator_sys_msg(paper_title: str, concept: str, concept_definition: str, concept_summary: str) -> tuple[SystemMessage, SystemMessage, SystemMessage]:
    '''Validator prompts for concept name, definition and summary (each answered with ResultValidation).'''
    approve_text = '''Set approved to true if it satisfies the objective. Otherwise set approved to false and explain briefly in reason what is wrong, so the next LLM can fix it.'''

    concept_validator_msg = SystemMessage(f'''You are a validator. Check if this extracted concept name is valid: "{concept}"
                        A valid {CONCEPT_NAME_PROMPT}
                        {approve_text}
                        ''')

    concept_definition_validator_msg = SystemMessage(f'''You are a semantic validator. Check if the following concept definition is correct: "{concept}" - "{concept_definition}"
        {DEFINITION_PROMPT}
        {approve_text}''')
    
    summary_validator_msg = SystemMessage(f'''You are a semantic validator. Check if the following summary is correct: "{concept_summary}"
        Your task is to check whether the extracted information satisfies the objective:
        
        Approve if ALL of these are true:
            1. The concept is defined factually and precisely and aligns with the concept "{concept}".
            2. {EXAMPLE_PROMPT}
            3. {USE_CASE_PROMPT} — and it references the paper title "{paper_title}".

        Do NOT reject based on formatting, sentence count, paragraph structure, or writing style.
        Only reject if the content is factually wrong, missing a section, or completely off-topic.

        {approve_text}''')

    return (concept_validator_msg, concept_definition_validator_msg, summary_validator_msg)

def create_paper_refiner_sys_msgs(validator_title_msg: str = "", validator_concept_msg: str = "", validator_concept_definition_msg: str = "", validator_summary_msg: str = "") -> tuple[SystemMessage, SystemMessage, SystemMessage, SystemMessage]:
    '''Refiner prompts for title, concept, definition and summary, each including the validator's feedback'''
    paper_title_msg = SystemMessage(f'''You are a research expert.
        From the list of available paper titles, choose the most appropriate one. Return ONLY the exact title, nothing else. {validator_title_msg}''')

    paper_concept_msg = SystemMessage(f'''You are given a concept name that was rejected by a validator. A correct {CONCEPT_NAME_PROMPT} Return ONLY the corrected name - The validator indicated: {validator_concept_msg}''')

    concept_definition_msg = SystemMessage(f'''Rewrite the given {DEFINITION_PROMPT} - The validator indicated: {validator_concept_definition_msg}''')

    concept_summary_msg = SystemMessage(
        f'''Rewrite the given summary using the paper abstract. Write exactly three parts, no introduction, no line breaks, no markdown:
        1. {SUMMARY_PROMPT}
        2. {EXAMPLE_PROMPT}
        3. {USE_CASE_PROMPT}
        - The validator indicated: {validator_summary_msg}''')
    
    return (paper_title_msg, paper_concept_msg, concept_definition_msg, concept_summary_msg)