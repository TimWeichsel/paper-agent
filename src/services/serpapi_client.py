import serpapi
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

def _transform_organic_results_to_dict(organic_results) -> dict:
    return {result["title"]: 
        {"summary": result["snippet"], 
        "author(s)": [author.get("name") for author in result.get("publication_info").get("authors")] if result.get("publication_info") and result.get("publication_info").get("authors") else [], 
        "pdf_link": result.get("resources")[0].get("link")} 
        for result in organic_results 
        if result.get("title") 
        and result.get("snippet") 
        and result.get("resources") 
        and result.get("resources")[0].get("file_format") == "PDF"}

@tool
def serpapi_paper_search(search_query: str , max_results: int = 30) -> dict:
    '''
    Calls the serpapi API with the given search query and returns a given snipped of the best found papers.
    Always use max_results=30. Maximum is 30.
    '''
    max_results = min(max_results, 30)
    
    client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))
    response = client.search({
        "engine": "google_scholar",
        "q": search_query,
        "hl": "en",
        "num": max_results
    })
    organic_results = response.get("organic_results")

    return _transform_organic_results_to_dict(organic_results) if organic_results else {"Error occured while using serpapi": "No organic results found in the API response."}

