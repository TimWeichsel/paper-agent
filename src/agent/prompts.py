from langchain_core.messages import SystemMessage

def create_paper_assistent_sys_msg(query_preference: str, complexity_preference: str = "medium", concept_base: str = "No concepts learned yet", knowledge_base: str = "No knowledge base found", tool_call_count: int = 0) -> SystemMessage:
    if tool_call_count >= 3:
        return SystemMessage(f'''You are a concept learning expert. Concepts the user already knows: {concept_base}. Knowledge base: {knowledge_base}.
    Complexity: "{complexity_preference}". Query: "{query_preference}".

    Using the search results above, return ONLY this structure:
    **Concept:** <concept name>
    **What it is:** <5-10 sentences: plain, precise definition — no examples, no analogies>
    **Connection to your knowledge:** <4 sentence: how this relates to what the user already knows>
    **Paper:** <title> — <authors, year>: <7 sentences on where and how this concept is concretely used or introduced in this paper>''')

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

def create_paper_organizer_sys_msg(validator_title_msg: str = "", validator_concept_msg: str = "", validator_summary_msg: str = "") -> tuple[SystemMessage, SystemMessage, SystemMessage]:
    if validator_title_msg != "":
        validator_title_msg = f"Note that the validator said noted an error in your last try: {validator_title_msg}"
    if validator_concept_msg != "":
        validator_concept_msg = f"Note that the validator said noted an error in your last try: {validator_concept_msg}"
    if validator_summary_msg != "":
        validator_summary_msg = f"Note that the validator said noted an error in your last try: {validator_summary_msg}"
    
    paper_titel_msg = SystemMessage(f'''You are a research expert.
        Your goal is to extract the title of the given paper information. Don't return anyhing but just the title. {validator_title_msg}''')

    paper_concept_msg = SystemMessage(f'''Extract the analyzed **Concept** of the given summary {validator_concept_msg}''')

    paper_summary_msg = SystemMessage(
        f'''From the following explanation, write exactly three things — no introduction, start directly:
        1. 4 sentences defining the concept precisely in plain terms. No examples, no analogies.
        2. Give an example how an expert would use this concept.
        3. One sentence: "Used in: [paper title] — [what role the concept plays in that paper]".  {validator_summary_msg}''')
    
    return (paper_titel_msg, paper_concept_msg, paper_summary_msg)

def create_validator_sys_msg(paper_title: str, paper_summary: str, paper_concept: str, last_human_message: str) -> tuple[SystemMessage, SystemMessage, SystemMessage]:
    paper_titel_msg, paper_concept_msg, paper_summary_msg =create_paper_organizer_sys_msg()
    title_validator_msg = SystemMessage(f'''You are a llm validator. The last llm's task was to extract the title from the following llm interaction:
                        {last_human_message}. The title is "{paper_title}".
                        Your task is to check wether the extracted information satisfies the objective: "{paper_titel_msg.content}". 
                        If it does return the word "approve" in your message (nothing else is needed but the following logic will check for this word).
                        If it does not satisfy the objective, return a message to explain the llm what was wrong (but do not use the key word "approve", otherwise it will be counted as correct)!
                        ''')
    
    concept_validator_msg = SystemMessage(f'''You are a llm validator. The last llm's task was to extract the concept from the following llm interaction:
                        {last_human_message}. The concept is "{paper_concept}".
                        Your task is to check wether the extracted information satisfies the objective: "{paper_concept_msg.content}". 
                        If it does return the word "approve" in your message (nothing else is needed but the following logic will check for this word).
                        If it does not satisfy the objective, return a message to explain the llm what was wrong (but do not use the key word "approve", otherwise it will be counted as correct)!
                        ''')
    summary_validator_msg = SystemMessage(f'''You are a semantic validator. Check if the following summary is correct: "{paper_summary}"
        Your task is to check wether the extracted information satisfies the objective: "{paper_concept_msg.content}"
        Approve if ALL of these are true:
            1. The concept is defined factually and precisely (does not need to be a specific number of sentences)
            2. A real-world usage example or application is included
            3. A paper title is referenced with "Used in:" and its role is described

        Do NOT reject based on formatting, sentence count, paragraph structure, or writing style.
        Only reject if the content is factually wrong, missing a section, or completely off-topic.

        If correct → include "approve" in your response.
        If not → explain specifically what content is wrong or missing (but do not use the key word "approve", otherwise it will be counted as correct)!.''')

    return (title_validator_msg, concept_validator_msg, summary_validator_msg)