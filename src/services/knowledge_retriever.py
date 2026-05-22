from utils.file_utils import load_file
from langchain_core.tools import tool

@tool
def load_paper_information():
    '''
    Loads the following information
        - current knowledge base of the user
        - all presented papers summarized
        - all titles of the presented papers 
    '''
    knowledge_base = load_file("src/data/knowledge_base.txt", "No knowledge yet")# Read knowledge_base.txt
    paper_base = load_file("src/data/paper_base.txt", "No papers analyzed yet")# Read paper_base.txt
    paper_list = load_file("src/data/paper_list.txt", "No papers analyzed yet")# Read paper_list.txt
    return {"knowledge_base": knowledge_base, "paper_base": paper_base, "paper_list": paper_list}