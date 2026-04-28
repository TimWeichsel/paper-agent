from dotenv import load_dotenv
load_dotenv()

from google import genai
import arxiv
from src.services.gemini_sdk import ask_gemini_with_retries
from utils.file_utils import load_file, save_file, append_to_file


def load_paper_informaion():
    knowledge_base = load_file("src/data/knowledge_base.txt", "No knowledge yet")# Read knowledge_base.txt
    paper_base = load_file("src/data/paper_base.txt", "No papers analyzed yet")# Read paper_base.txt
    paper_list = load_file("src/data/paper_list.txt", "No papers analyzed yet")# Read paper_list.txt
    return knowledge_base, paper_base, paper_list


def ask_api_print_response(prompt: str, model: str = "gemini", topic: str = "API Call"):
    match model:
        case "gemini":
            response = ask_gemini_with_retries(prompt)
            print(f"##### {topic} ######")
            print(response)
            print(f"###################################################")
            return response
        case _:
            raise NotImplementedError(f"Model {model} not implemented")

def arxiv_paper_call(query: str, topic: str = "Arxiv Call", max_results: int = 30):
    papers = list(arxiv.Search(query=query, max_results=max_results).results())
    print(f"##### {topic} ######")
    for paper in papers:
        print(paper.title)
    print(f"###################################################")
    return papers

def find_first_paper_by_title(papers: list, title: str):
    for paper in papers:
        if paper.title in title or title in paper.title:
            return paper
    return None

def summarize_information(paper_explaination,new_paper_base):
    explaination_prompt = f'''Please summarize the following paper to easy to unerstand 200 words: \"{paper_explaination}\"'''
    new_paper_base_prompt = f'''Please summarize the following knowledge base to easy to unerstand 200 words: \"{new_paper_base}\"'''
    explaination_summary = ask_api_print_response(explaination_prompt, "gemini", "Explaination Summary")
    new_paper_base_summary = ask_api_print_response(new_paper_base_prompt, "gemini", "New Paper Base Summary")
    return explaination_summary, new_paper_base_summary

def update_knowledge_with_new_paper(add_on_input: str = None, summarize=True):
    knowledge_base, paper_base, paper_list = load_paper_informaion()      
    # Key Word Search
    if add_on_input:
        add_on_input = f" Specifically I am interested in learning the conept(s)/content(s): \"{add_on_input}\"."
    learn_prompt= f"""I want to learn more about ML. 
    I might already have some knowledge which is stored in the following text: \"{knowledge_base}\". 
    Also, I might have learned already about the following papers: \"{paper_list}\" 
    and received this information, here a summary: \"{paper_base}\".  
    Please find a paper that extends good my current knowledge. 
    For that I have the arxiv access. So the only output I need from you is some connected keywords 
    for an arxiv paper seach. Please only give me 3-5 words that should find a good paper 
    that extends my knowledge and perfectly adds to my previous knowledge. 
    No markdown formatting, just plain text.{add_on_input}"""
    key_word_text = ask_api_print_response(learn_prompt, "gemini", "Keyword Search")

    # Paper Search
    papers = arxiv_paper_call(key_word_text, "Arxiv Paper Search", max_results=15)
    paper_titels = "\n".join([p.title for p in papers])


    # Paper Decision
    paper_contents= f"""I want you to choose a paper that extend my current knowledge of Machine Learning
    , which is described by the following text:\"{knowledge_base}\".
    {add_on_input}
    My knowledge was extendet by the following papers: \"{paper_list}\" and the following information: \"{paper_base}\" in the past.
    Please return only the exact title of the paper and nothing else! 
    Here are all possible papers I found {paper_titels} """
    paper_decision = ask_api_print_response(paper_contents, topic="Paper Decision")


    # Find Paper
    paper = find_first_paper_by_title(papers, paper_decision)
    if paper is not None:
        print(f"Found the paper: {paper.title} with the following abstract: {paper.summary} as new learning material")
    else:
        print("No paper found")
        exit()

    # Explain the paper in a simple but very detailed way
    paper_summary = f"""I have the following knowledge about Machine Learning: "{knowledge_base}".
    I found a paper with the title "{paper.title}" and the following abstract: "{paper.summary}".

    {add_on_input}
    Please explain this paper in the following strict structure:

    PART 1 - DEEP FOUNDATION (this is the most important part, take a lot of space):
    Before explaining the paper itself, teach me all the foundational concepts 
    I need to truly understand it. Go very deep here. Assume I know the basics 
    from my knowledge base but want to understand them on a much deeper level.
    For every concept:
    - Explain the intuition simply first
    - Then go deeper mathematically or technically  
    - Give a concrete example
    - Explain WHY it works, not just WHAT it does
    - Connect it to things I already know from my knowledge base

    PART 2 - THE PAPER ITSELF:
    Now explain what problem this paper solves and why existing solutions 
    were not good enough. Then explain the core innovation step by step.
    Use the foundation from Part 1 to build on.

    PART 3 - THE KEY INSIGHT (one paragraph):
    What is the single most important thing to remember from this paper?

    PART 4 - MODERN RELEVANCE:
    Where is this used today? Which models use it? 
    Why does it matter for understanding modern ML?

    Be thorough in Part 1 — I want to deeply understand the prerequisites,
    not just skim them."""

    paper_explaination = ask_api_print_response(paper_summary, topic="Paper Summary")


    knowledge_prompt = f"""I have the following knowledge about Machine Learning: 
    \"{paper_base}\" (paper base). I received a paper with the title \"{paper.title}\"
    and the following abstract: \"{paper.summary}\". 
    I got the following explanation of the paper: \"{paper_explaination}\".
    {add_on_input} 
    Please update my current knowledge by including this new information, 
    if my knowledge is already very detailed this new information should not take to much space. 
    Else, it can take more space. I want a holistic knowledge base, considering both!"""

    new_paper_base = ask_api_print_response(knowledge_prompt, topic="New Paper Base")

    # Update the paper_base.txt with the current knowledge
    save_file("src/data/paper_base.txt", new_paper_base)

    # Update the paper_list.txt with the current title
    append_to_file("src/data/paper_list.txt", paper.title + "\n")

    if summarize:
        paper_explaination, new_paper_base = summarize_information(paper_explaination,new_paper_base)

    return {
        "title":       paper.title,
        "explanation": paper_explaination,
        "paper_base":  new_paper_base
    }
