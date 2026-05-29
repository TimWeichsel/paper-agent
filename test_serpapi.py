import serpapi
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

def _transform_organic_results_to_dict(organic_results) -> dict:
    return {result["title"]: result["snippet"] for result in organic_results if result.get("title") and result.get("snippet")}

#@tool
def serpapi_paper_search(search_query: str , max_results: int = 20) -> dict:
    '''
    Calls the serpapi API with the given search query and returns a given snipped of the best found papers.
    Always use max_results=20. Maximum is 20.
    '''
    print(os.getenv("SERPAPI_API_KEY"))
    client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))
    response = client.search({
        "engine": "google_scholar",
        "q": search_query,
        "hl": "en",
        "num": max_results
    })
    organic_results = response.get("organic_results")
    return _transform_organic_results_to_dict(organic_results) if organic_results else {"Error occured while using serpapi": "No organic results found in the API response."}

serpapi_paper_search(search_query="machine learning for healthcare", max_results=5)