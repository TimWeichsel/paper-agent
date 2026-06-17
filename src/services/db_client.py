import pgvector
import psycopg2 #postgres connection library
from pgvector.psycopg2 import register_vector #gives psycopg2 the ability to handle vector data types
from psycopg2 import OperationalError
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()

host = os.getenv("POSTGRES_HOST", "localhost")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
dbname = os.getenv("POSTGRES_DB")

embedding_dimension = 3072

def _get_embedding(content: str) -> int:
    client = genai.Client()
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=content,
        config=types.EmbedContentConfig(output_dimensionality=embedding_dimension)
    )
    return result.embeddings[0].values


def _connect_to_db():
    try:
        connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=5432)
        return connection
    except OperationalError as e:
        print(f"Error connecting to PostgreSQL database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def initialize_db():
    connection = _connect_to_db()
    if connection is None:
        print("Failed to connect to db.")
        return None
    cursor = connection.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS papers
        (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        abstract TEXT,
        pdf_link TEXT,
        publication_date DATE,
        citation_count INTEGER,
        embedding vector({embedding_dimension})                  
        );
                   
        CREATE TABLE IF NOT EXISTS concepts
        (
        id SERIAL PRIMARY KEY,
        paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
        concept TEXT NOT NULL,
        concept_summary TEXT,
        embedding vector({embedding_dimension})         
        );     

        CREATE TABLE IF NOT EXISTS authors
        (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
        );
                   
        CREATE TABLE IF NOT EXISTS paper_authors
        (
        paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
        author_id INTEGER REFERENCES authors(id) ON DELETE CASCADE,
        PRIMARY KEY (paper_id, author_id)    
        );
    """)
    connection.commit()
    cursor.close()
    connection.close()

def drop_db():
    connection = _connect_to_db()
    if connection is None:
        print("Failed to connect to db.")
        return None
    cursor = connection.cursor()
    cursor.execute("""
        DROP TABLE IF EXISTS paper_authors;
        DROP TABLE IF EXISTS concepts;
        DROP TABLE IF EXISTS authors;
        DROP TABLE IF EXISTS papers;
    """)
    connection.commit()
    cursor.close()
    connection.close()


def _insert_authors_to_db(connection, authors: list[str]) -> list[int]:
    cursor = connection.cursor()
    ids = []
    for author in authors:
        cursor.execute("INSERT INTO authors (name) VALUES (%s) RETURNING id", (author,))
        ids.append(cursor.fetchone()[0])
    connection.commit()
    cursor.close()
    return ids

def _insert_concept_to_db(connection, paper_id: int, concept: str, concept_summary: str) -> int:
    cursor = connection.cursor()
    embedding = _get_embedding(concept)
    cursor.execute("INSERT INTO concepts (paper_id, concept, concept_summary, embedding) VALUES (%s, %s, %s, %s) RETURNING id", (paper_id, concept, concept_summary, embedding))
    concept_id = cursor.fetchone()[0]
    connection.commit()
    cursor.close()
    return concept_id

def _insert_paper_to_db(connection, title: str, abstract: str, pdf_link: str, publication_date: str, citation_count: int) -> int:
    cursor = connection.cursor()
    embedding = _get_embedding(abstract)
    cursor.execute("INSERT INTO papers (title, abstract, pdf_link, publication_date, citation_count, embedding) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id", (title, abstract, pdf_link, publication_date, citation_count, embedding))
    paper_id = cursor.fetchone()[0]
    connection.commit()
    cursor.close()
    return paper_id

def _connect_author_to_paper(connection, paper_id: str, author_id: str):
    cursor = connection.cursor()
    cursor.execute("INSERT INTO paper_authors (paper_id, author_id) VALUES (%s, %s)", (paper_id, author_id))
    connection.commit()
    cursor.close()

def add_paper_with_concepts_to_db(paper_title: str, paper_abstract: str, paper_pdf_link: str, paper_publication_date: str, paper_citation_count: int, concept: str, concept_summary: str, authors: list[str]):
    connection = _connect_to_db()
    register_vector(connection)  #Vector Converter is added to connction
    author_ids = _insert_authors_to_db(connection, authors)
    paper_id = _insert_paper_to_db(connection, paper_title, paper_abstract, paper_pdf_link, paper_publication_date, paper_citation_count)
    for author_id in author_ids:
        _connect_author_to_paper(connection, paper_id, author_id)
    concept_ids = _insert_concept_to_db(connection, paper_id, concept, concept_summary)
    connection.close()
    return paper_id, concept_ids, author_ids


if __name__ == "__main__":
    '''
    initialize_db()
    paper_id, concept_ids, author_ids = add_paper_with_concepts_to_db(
        paper_title="Sample Paper Title",
        paper_abstract="This is a sample abstract for the paper.",
        paper_pdf_link="http://example.com/sample.pdf",
        paper_publication_date="2024-01-01",
        paper_citation_count=10,
        concept="Sample Concept",
        concept_summary="This is a summary of the sample concept.",
        authors=["Author One", "Author Two"])
    print(f"Inserted paper with ID: {paper_id}, concept IDs: {concept_ids}, author IDs: {author_ids}")
    '''
    drop_db()
    