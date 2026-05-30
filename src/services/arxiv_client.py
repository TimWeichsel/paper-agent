import arxiv
from langchain_core.tools import tool


@tool
def arxiv_paper_search(search_query: str , max_results: int = 30) -> dict:
    '''
    Calls the arxiv API with the given search query and returns the abstracts of the best found papers.
    Always use max_results=30. Maximum is 30.
    '''
    print("arxiv tool called with query:", search_query, "and max_results:", max_results)
    if max_results > 10:
        max_results = 10
    try:
        papers = list(arxiv.Client(delay_seconds=3, page_size=30).results(arxiv.Search(query=search_query, max_results=int(max_results))))
        abstracts = {paper.title: paper.summary for paper in papers}
    except Exception as e:
        return {"ERROR": f"API failed ({str(e)}). Do NOT suggest any paper. Tell the user the search failed and they should try again."}
    return abstracts