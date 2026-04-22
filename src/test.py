from dotenv import load_dotenv
load_dotenv()

from google import genai
import arxiv
from services.gemini_sdk import ask_gemini_with_retries

content = "No Knowledge yet"
# Read basis.txt
with open("data/basis.txt", "r") as file:
    content = file.read()

# API Gemini Client
learn_prompt= f"I want to learn more about ML. I might already have some knowledge which is stored in the following text: \"{content}\". Please find a paper that extends good my current knowledge. For that I have the arxiv access. So the only output I need from you is some connected keywords for an arxiv paper seach. Please only give me 3-5 words that should find a good paper that extends my knowledge. No markdown formatting, just plain text. "
key_word_text = ask_gemini_with_retries(learn_prompt)
print(key_word_text)

# Paper Search
paper_results = arxiv.Search(query=key_word_text)

for paper in paper_results.results():
    print(paper.title)
    
# Paper Decision
paper_contents= f"I want you to choose a paper that extend my current knowledge of Machine Learning, which is described by the following text(might be empty):\"{content}\". Please return only the exact title of the paper and nothing else! Here are all possible papers I found {paper_results} "
paper_decision = ask_gemini_with_retries(paper_contents)
print(paper_decision)


title = None
abstract = None
# Find Paper
for paper in paper_results.results():
    if paper_decision in paper.title or \
       paper.title in paper_decision:
        title = paper.title
        abstract = paper.summary
        print(f"Found the paper: {title} with the following abstract: {abstract}")
        break

# Explain the paper in a simple but very detailed way
paper_summary = f"I have the following knowledge about Machine Learning: \"{content}\". I found a paper with the title \"{title}\" and the following abstract: \"{abstract}\". Please explain the content of the paper in a simple but very detailed way, so that I can understand it and learn from it. This should extend my current knowledge and not repeat it! "
paper_explaination = ask_gemini_with_retries(paper_summary)
print(paper_explaination)


knowledge_prompt = f"I have the following knowledge about Machine Learning: \"{content}\". I found a paper with the title \"{title}\" and the following abstract: \"{abstract}\". I got the following explanation of the paper: \"{paper_explaination.text}\". Please update my current knowledge by including tthis new information, if my knowledge is already very detailied this new information should not take to much space. Else, it can take more space. I want aa holistic knowledge base! "
new_knowledge_answer = ask_gemini_with_retries(knowledge_prompt)
print(new_knowledge_answer)


# Update the basis.txt with the current knowledge
with open("basis.txt", "w") as file:
    file.write(new_knowledge_answer)
