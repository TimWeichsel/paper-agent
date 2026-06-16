import pgvector
import psycopg2 #postgres connection library
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
    return result.embeddings[0]


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
        name TEXT NOT NULL
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



if __name__ == "__main__":
    initialize_db()
    connection = _connect_to_db()
    ids = _insert_authors_to_db(connection, ["Author 1", "Author 2", "Author 3"])
    print(ids)
    connection.close()
    