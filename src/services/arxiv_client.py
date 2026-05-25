import arxiv
from langchain_core.tools import tool


@tool
def arxiv_paper_search(search_query: str , max_results: int = 10) -> dict:
    '''
    Calls the arxiv API with the given search query and returns the abstracts of the best found papers.
    '''
    try:
        papers = list(arxiv.Client(delay_seconds=1).results(arxiv.Search(query=search_query, max_results=int(max_results))))
        abstracts = {paper.title: paper.summary for paper in papers}
    except Exception as e:
        abstracts = {"Error occured while using arxiv API": str(e)}
    return abstracts