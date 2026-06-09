from langchain_core.messages import SystemMessage

def create_paper_assistent_sys_msg(query_preference: str, complexity_preference: str = "medium", concept_names: str = "No concepts learned yet", research_goals: str = "No research goals found", tool_call_count: int = 0, learning_mode: str = "new") -> SystemMessage:
    if tool_call_count >= 3:
        return SystemMessage(f'''You are a concept learning expert.
    Use the concept you already identified and searched for earlier in this conversation. Do NOT switch to a different concept.
    Using the search results above, return ONLY this structure:
    **Concept:** <concept name> | **What it is:** <5-10 sentences: plain, precise definition — no examples, no analogies> | **Connection to your knowledge:** <4 sentence: how this relates to what the user already knows> | **Paper:** <title> — <authors, year>: <7 sentences on where and how this concept is concretely used or introduced in this paper>
    Include the structure:  **<part>**: ... | ''')

    if learning_mode == "new":
        mode_instruction = "Pick a concept the user has NOT learned yet — it must be completely new and have a different name than any concept listed above."
    else:
        mode_instruction = "Pick a concept in the same field as one of the concepts above, but a genuinely different concept — not a subcategory, variation, or rephrasing of an already-learned concept."

    return SystemMessage(f'''You are a research assistant.
    User research goals: {research_goals}
    Concepts already learned (do NOT pick these or any variation of them): {concept_names}
    Instruction: {mode_instruction}
    Complexity preference: "{complexity_preference}"
    Query: "{query_preference}"

    Identify the next concept the user should learn. It must be a standalone, well-known concept — NOT a broad field or topic cluster.
    Then find the canonical paper and call semantic_scholar_paper_search (or serpapi_paper_search if the topic is suitable and semantic_scholar_paper_search is not working) up to three times.
    If a paper has no PDF or results are irrelevant, try a different paper for the same concept or use serpapi_paper_search as fallback. Prefer papers with open access PDFs.
    Do NOT use the raw user query as the search term — derive the actual paper title first.''')

def create_paper_organizer_sys_msg(validator_title_msg: str = "", validator_concept_msg: str = "", validator_summary_msg: str = "") -> tuple[SystemMessage, SystemMessage, SystemMessage]:
    if validator_title_msg != "":
        validator_title_msg = f"Note that the validator said noted an error in your last try: {validator_title_msg}"
    if validator_concept_msg != "":
        validator_concept_msg = f"Note that the validator said noted an error in your last try: {validator_concept_msg}"
    if validator_summary_msg != "":
        validator_summary_msg = f"Note that the validator said noted an error in your last try: {validator_summary_msg}"
    
    paper_titel_msg = SystemMessage(f'''You are a research expert.
        Your goal is to extract the title of the given paper information. Don't return anyhing but just the title. {validator_title_msg}''')

    paper_concept_msg = SystemMessage(f'''Extract the concept name from the given summary. Return ONLY the concept name — no formatting, no "The analyzed Concept is:", no markdown bold. Just the plain name. {validator_concept_msg}''')


    paper_summary_msg = SystemMessage(
        f'''From the following explanation, write exactly three things — no introduction, start directly and don't use line breaks or markdown formatting:
        1. 4 sentences defining the concept precisely in plain terms. No examples, no analogies.
        2. Give an example how an expert would use this concept and how it would be applied in research y given details to the concept!
        3. One sentence: "Used in: [paper title] — [what role the concept plays in that paper]".  {validator_summary_msg}''')
    
    return (paper_titel_msg, paper_concept_msg, paper_summary_msg)

def create_validator_sys_msg(paper_title: str, paper_summary: str, paper_concept: str) -> tuple[SystemMessage, SystemMessage, SystemMessage]:
    #paper_titel_msg, paper_concept_msg, _ =create_paper_organizer_sys_msg()
    approve_text = f"If it does return the word \"approve\" in your very short message (nothing else is needed but the following logic will check for this word). If it does not satisfy the objective, return also a short message to explain the llm what was wrong (but do not use the key word \"approve\", otherwise it will be counted as correct)!"
    
    title_validator_msg = SystemMessage(f'''You are a validator. Check if this extracted title is valid: "{paper_title}"
                        A valid title is a plain text string — no extra explanation, no formatting.
                        {approve_text}
                        ''')

    concept_validator_msg = SystemMessage(f'''You are a validator. Check if this extracted concept name is valid: "{paper_concept}"
                        A valid concept name is a short, plain text string — no extra explanation, no formatting.
                        {approve_text}
                        ''')
    
    summary_validator_msg = SystemMessage(f'''You are a semantic validator. Check if the following summary is correct: "{paper_summary}"
        Your task is to check wether the extracted information satisfies the objective:
        
        Approve if ALL of these are true:
            1. The concept is defined factually and precisely (does not need to be a specific number of sentences)
            2. An example how an expert would use this concept and how it would be applied in research is included with details.
            3. A paper title is referenced with "Used in:" and its role is described

        Do NOT reject based on formatting, sentence count, paragraph structure, or writing style.
        Only reject if the content is factually wrong, missing a section, or completely off-topic.

        {approve_text}''')

    return (title_validator_msg, concept_validator_msg, summary_validator_msg)