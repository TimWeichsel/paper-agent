import arxiv

client = arxiv.Client(delay_seconds=3)
search = arxiv.Search(query="neural networks", max_results=3)

try:
    papers = list(client.results(search))
    print(f"Success! Found {len(papers)} papers:")
    for paper in papers:
        print(f"  - {paper.title}")
except Exception as e:
    print(f"Failed: {e}")
