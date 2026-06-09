import requests
import json
from langchain_core.tools import tool
import os
import time
from dotenv import load_dotenv

load_dotenv()

def _transform_semantic_scholar_results_to_dict(json_data: list) -> dict:
    return {
        paper["title"]: {
                "summary": paper.get("abstract"), 
                "author(s)": [author["name"] for author in paper.get("authors", [])], 
                "date": paper.get("publicationDate"),
                "citation_count": paper.get("citationCount", 0),
                "pdf_link": paper.get("openAccessPdf").get("url"),
        }
        for paper in json_data 
        if paper.get("title") 
        and paper.get("abstract")
        and paper.get("authors")
        and paper.get("publicationDate")
        and paper.get("openAccessPdf")
        and paper.get("openAccessPdf").get("url")}

@tool
def semantic_scholar_paper_search(search_query: str , results: int = 10) -> dict:
    '''
    Calls the semantic scholar API with the given search query and returns title, abstract, authors, date, citation count, and PDF link of the best found papers (only papers with pdf links are included).
    Search query needs to be specified to get results in the domain of interest. Results specifies how many results should be returned (minimum of 5, maximum of 10).
    '''
    results = max(min(results, 10), 5)
    new_max_results = results

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    url = "http://api.semanticscholar.org/graph/v1/paper/search"
    query_parmams = {
        "query": search_query,
        "fields": "title,authors,abstract,citationCount,publicationDate,openAccessPdf",
        "limit": str(results),
    }

    headers = {"x-api-key": api_key}

    response = requests.get(url, params=query_parmams, headers=headers)

    if response.status_code == 200:
        transformed_response = _transform_semantic_scholar_results_to_dict(response.json()["data"])
    elif response.status_code == 429:
        time.sleep(10)
        transformed_response = {}
    else:
        return {"ERROR": f"API call failed with error status code {response.status_code}."}

    while len(transformed_response) < results and new_max_results < 100:
        time.sleep(2)
        new_max_results = new_max_results + results
        query_parmams = {
        "query": search_query,
        "fields": "title,authors,abstract,citationCount,publicationDate,openAccessPdf",
        "limit": str(new_max_results),
        }
        response = requests.get(url, params=query_parmams, headers=headers)
        if response.status_code == 200:
            transformed_response.update(_transform_semantic_scholar_results_to_dict(response.json()["data"]))
        elif response.status_code == 429:
            time.sleep(3)
        else:
            if transformed_response:
                return transformed_response
            return {"ERROR": f"API call failed with error status code {response.status_code}."}

    return dict(list(transformed_response.items())[:results]) #slice dict