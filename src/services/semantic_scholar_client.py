import requests
from langchain_core.tools import tool
import os
import time
from dotenv import load_dotenv
import json

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

load_dotenv()
STATUS_CODE_429_WARNING = 429
STATUS_CODE_200_SUCCESS = 200
MIN_CITATION_COUNT = 20
ABSTRACT_MAX_LENGTH = 250


@tool(response_format="content_and_artifact") #content for LLMs and artifact for db and embeddings
def semantic_scholar_paper_search(concept: str, search_query: str , results: int = 5) -> tuple[str, dict]:
    '''
    Searches Semantic Scholar for papers about the given concept and returns title, year, citation count,
    first author and a short abstract of the best found papers.
    Parameters:
    - concept: the canonical name of the concept as used in the literature (not the user's wording); keep it identical in all searches
    - search_query: ONLY the exact title of the paper (e.g. "Some methods for classification and analysis of multivariate observations"). Never add author names or years.
    - results: number of results (minimum 5, maximum 10)
    '''
    results = max(min(results, 10), 5)

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    url_broad_search = "https://api.semanticscholar.org/graph/v1/paper/search"
    url_match_search = "https://api.semanticscholar.org/graph/v1/paper/search/match"

    query_params = {
        "query": search_query,
        "fields": "title,authors,abstract,citationCount,publicationDate,openAccessPdf,url",
        "limit": str(results),
        "minCitationCount": MIN_CITATION_COUNT
    }

    headers = {"x-api-key": api_key}

    #First direct search

    response_match = _search_with_retries(url=url_match_search, params={"query": search_query, "fields": query_params["fields"]}, headers=headers)
    if response_match.status_code == STATUS_CODE_200_SUCCESS:
        match_papers = _transform_semantic_scholar_results_to_dict(response_match.json().get("data", []))
        match_papers = {title: paper for title, paper in match_papers.items() if paper["citation_count"] >= MIN_CITATION_COUNT}
    else:
        match_papers = {}

    #Second broad search
    response_broad = _search_with_retries(url=url_broad_search, params=query_params, headers=headers)
    broad_papers = _transform_semantic_scholar_results_to_dict(response_broad.json().get("data", [])) if response_broad.status_code == STATUS_CODE_200_SUCCESS else {}

    papers = {**match_papers, **broad_papers}
    logger.debug(f"semantic scholar: concept='{concept}', query='{search_query}', limit={results}, found={len(papers)}, match_status={response_match.status_code}, match={len(match_papers)}, broad={len(broad_papers)}")
    if not papers:
        return f"ERROR: No well-cited papers found. Try a different paper (e.g. a survey or tutorial) for the SAME concept.", {}
    return _minimize_paper_info_to_text(papers), papers

def _search_with_retries(url, params:dict, headers:dict) -> requests.Response:
    time.sleep(1)
    response = requests.get(url=url, params=params, headers=headers, timeout=10)
    if response.status_code == STATUS_CODE_429_WARNING:
        time.sleep(5)
        response = requests.get(url=url, params=params, headers=headers, timeout=10)
    return response

def _transform_semantic_scholar_results_to_dict(json_data: list) -> dict:
    return {
        paper["title"]: {
                "abstract": paper.get("abstract") or "", 
                "author(s)": [author["name"] for author in paper.get("authors", [])], 
                "date": paper.get("publicationDate"),
                "citation_count": paper.get("citationCount", 0),
                "pdf_link": (paper.get("openAccessPdf") or {}).get("url") or paper.get("url", ""),
        }
        for paper in json_data 
        if paper.get("title") 
        and paper.get("authors")
        and paper.get("publicationDate")}

def _minimize_paper_info_to_text(papers: dict) -> str:
    paper_minimized = {
        title: {
            "year": paper["date"][:4],
            "citation_count": paper["citation_count"],
            "abstract": (paper["abstract"][:ABSTRACT_MAX_LENGTH] + "..." if len(paper["abstract"]) > ABSTRACT_MAX_LENGTH else paper["abstract"]) or "No abstract available",
            "first_author": paper["author(s)"][0] if paper["author(s)"] else "Unknown"
        } 
        for title, paper in papers.items()
    }
    return json.dumps(paper_minimized)