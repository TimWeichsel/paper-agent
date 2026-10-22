# Paper Agent by TimWeichselberger
This project should help students learning different concepts in their domain. By defining their research goal, they can learn new concepts each backed with a relevant paper (to further study if interested) found with semantic scholar. All studied concepts are saved and visualized (PCA plot).

## 1 How it works
![Agent Graph](static_images/graph.png)

This RAG project consists of three agents: Paper-Researcher, Validator and Paper-Refiner

## 2 Tech Stack
- Python (3.12.11)
    - LangGraph for RAG
    - pgvector for VectorDB
    - Gradio as frontend
- Docker
- Conda (Optional)

## 3 Requirements for .env File (see .env.example)
- API-KEYS:
    - GROQ_API_KEY (free tier is sufficient)
    - GOOGLE_API_KEY (free tier is sufficient)
    - SEMANTIC_SCHOLAR_API_KEY (free tier is sufficient)

- DB Information
    - POSTGRES_USER=
    - POSTGRES_PASSWORD=
    - POSTGRES_DB=

## 4 Python Requirements (and Conda)

requirements.txt includes all needed python packages for this installation. One way is to install a Conda environment with Python = 3.12.11:

```bash
conda create -n paper-agent python=3.12.11
conda activate paper-agent
pip install -r requirements.txt
```

## 5 Docker Requirements

## 6 Start Project
- git clone the project
- cp .env.example .env
- docker compose up --build
- open http://localhost:7860

## 7 Usage
4 Tabs to use:
- Learn Concept - study a new concept by indicating search term, complexity preference, learning mode (complete new concept or related concept)
- Current Concept Base - See all so far studied concepts with their summary
- Current Paper List - All papers of the studied concepts
- Current Research Goals - Define your Research Goals so the agents find the right papers and concepts for you to stuy

## Limitations
- project runs in your ocker container and thus only via localhost
- rate-limits for free tier LLM plans might be nott sufficient
- LLM models might be depricated

