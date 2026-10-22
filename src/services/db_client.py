import psycopg2 #postgres connection library
from pgvector.psycopg2 import register_vector #gives psycopg2 the ability to handle vector data types
from psycopg2 import OperationalError
from dotenv import load_dotenv
import numpy as np
import os
from google import genai
from google.genai import types
load_dotenv()

host = os.getenv("POSTGRES_HOST", "localhost") #Fallback to localhost if POSTGRES is not initialized by Docker Compose (e.g. local app testing)
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
dbname = os.getenv("POSTGRES_DB")

import logging
logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 3072
CLIENT = genai.Client()


#DB management
def _connect_to_db():
    try:
        connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=5432)
        cursor = connection.cursor()
        return connection, cursor
    except OperationalError as e:
        logger.error(f"Error connecting to PostgreSQL database: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return None, None

def initialize_db():
    '''Creates the database tables with pgvector extension, if they do not exist yet.'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS papers
        (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL UNIQUE,
        abstract TEXT,
        pdf_link TEXT,
        publication_date DATE,
        citation_count INTEGER,
        embedding vector({EMBEDDING_DIMENSION})                  
        );
                   
        CREATE TABLE IF NOT EXISTS concepts
        (
        id SERIAL PRIMARY KEY,
        paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
        concept TEXT NOT NULL,
        concept_definition TEXT,
        concept_summary TEXT,
        embedding vector({EMBEDDING_DIMENSION})         
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

        CREATE TABLE IF NOT EXISTS research_goals
        (
        id INTEGER PRIMARY KEY DEFAULT 1,
        goals TEXT
        );
        INSERT INTO research_goals (id, goals) VALUES (1, '') ON CONFLICT (id) DO NOTHING;

    """)
    connection.commit()
    cursor.close()
    connection.close()

def drop_db():
    '''Drops all tables except the research goal table.'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return
    cursor.execute("""
        DROP TABLE IF EXISTS paper_authors;
        DROP TABLE IF EXISTS concepts;
        DROP TABLE IF EXISTS authors;
        DROP TABLE IF EXISTS papers;
    """)
    connection.commit()
    cursor.close()
    connection.close()

def drop_concept_table():
    '''Drops only the concept table and its relations to papers and authors.'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return
    cursor.execute("""
        DROP TABLE IF EXISTS concepts;
    """)
    connection.commit()
    cursor.close()
    connection.close()

#Setter
def add_paper_with_concepts_to_db(paper_title: str, paper_abstract: str, paper_pdf_link: str, paper_publication_date: str, paper_citation_count: int, concept: str, concept_summary: str, concept_definition: str, authors: list[str]) -> bool:
    '''Adds paper information with its authors and concepts to the database. Either all information is added or none (transaction).'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return False
    try:
        register_vector(connection)  #Vector Converter is added to connection
        author_ids = _insert_authors_to_db(cursor=cursor, authors=authors)
        paper_id = _insert_paper_to_db(cursor=cursor, title=paper_title, abstract=paper_abstract, pdf_link=paper_pdf_link, publication_date=paper_publication_date, citation_count=paper_citation_count)
        for author_id in author_ids:
            _connect_author_to_paper(cursor=cursor, paper_id=paper_id, author_id=author_id)
        _insert_concept_to_db(cursor=cursor, paper_id=paper_id, concept=concept, concept_definition=concept_definition, concept_summary=concept_summary)
        connection.commit() # commit complete transaction
        return True
    except Exception as e:
        logger.error(f"Error adding paper with concept to db: {e}")
        connection.rollback() # rollback transaction
        return False
    finally:
        cursor.close()
        connection.close()


def update_research_goals_in_db(new_goals: str):
    '''Updates the stored research goal in the database.'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return False
    cursor.execute("""
        UPDATE research_goals SET goals = %s WHERE id = 1
    """, (new_goals,))
    connection.commit()
    cursor.close()
    connection.close()

def _insert_authors_to_db(cursor: object, authors: list[str]) -> list[int]:
    ids = []
    for author in authors:
        cursor.execute("INSERT INTO authors (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (author,))
        cursor.execute("SELECT id FROM authors WHERE name = %s", (author,))
        ids.append(cursor.fetchone()[0])
    return ids

def _insert_concept_to_db(cursor: object, paper_id: int, concept: str, concept_definition: str, concept_summary: str) -> int:
    embedding = _get_embedding(f"{concept}: {concept_definition}")
    cursor.execute("INSERT INTO concepts (paper_id, concept, concept_definition, concept_summary, embedding) VALUES (%s, %s, %s, %s, %s) RETURNING id", (paper_id, concept, concept_definition, concept_summary, embedding))
    concept_id = cursor.fetchone()[0]
    return concept_id

def _insert_paper_to_db(cursor: object, title: str, abstract: str, pdf_link: str, publication_date: str, citation_count: int) -> int:
    embedding = _get_embedding(abstract or title)
    cursor.execute("INSERT INTO papers (title, abstract, pdf_link, publication_date, citation_count, embedding) " \
    "VALUES (%s, %s, %s, %s, %s, %s) " \
    "ON CONFLICT (title) DO UPDATE SET title = EXCLUDED.title " \
    "RETURNING id", (title, abstract, pdf_link, publication_date, citation_count, embedding))
    paper_id = cursor.fetchone()[0]
    return paper_id

def _connect_author_to_paper(cursor: object, paper_id: int, author_id: int):
    cursor.execute("INSERT INTO paper_authors (paper_id, author_id) " \
    "VALUES (%s, %s) " \
    "ON CONFLICT DO NOTHING", (paper_id, author_id))

# Getter
def get_all_concepts_from_db():
    '''Retrieves all concept names with their definitions and summaries from the database.'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return []
    cursor.execute("SELECT concept, concept_definition, concept_summary FROM concepts ORDER BY id")
    results = cursor.fetchall()
    concepts = [f"{concept}\n {concept_definition}\n {concept_summary}\n\n" for concept, concept_definition, concept_summary in results]
    cursor.close()
    connection.close()
    return concepts

def get_all_concept_names_from_db():
    '''Retrieves all concept names from the database.'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return []
    cursor.execute("SELECT concept FROM concepts ORDER BY id")
    results = cursor.fetchall()
    concept_names = [concept for concept, in results]
    cursor.close()
    connection.close()
    return concept_names

def get_all_paper_titles_from_db():
    '''Retrieves all paper titles from the database.'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return []
    cursor.execute("SELECT title FROM papers ORDER BY id")
    results = cursor.fetchall()
    paper_titles = [title for title, in results]
    cursor.close()
    connection.close()
    return paper_titles

def get_concepts_with_embeddings():
    '''Retrieves all concept names and their embeddings from the database.'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return []
    register_vector(connection)
    cursor.execute("SELECT concept, embedding FROM concepts ORDER BY id")
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return results

def get_research_goal_from_db() -> str:
    '''Retrieves the research goal from the database.'''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return "No research goals found."
    cursor.execute("SELECT goals FROM research_goals WHERE id = 1")
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result[0] if result and result[0] else "No research goals found."

def get_most_similar_concept(concept: str, concept_definition: str) -> tuple[str, float] | None:
    '''Retrieves the most similar concept with its cosine distance of the given concept and its definition from the database. '''
    connection, cursor = _connect_to_db()
    if connection is None or cursor is None:
        return None
    register_vector(connection)
    embedding = np.array(_get_embedding(f"{concept}: {concept_definition}")) #same embedding creation definition as in _insert_concept_to_db! 
    cursor.execute("SELECT concept, embedding <=> %s AS cosine_distance FROM concepts ORDER BY cosine_distance LIMIT 1", (embedding,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result


def _get_embedding(content: str) -> list[float]:
    result = CLIENT.models.embed_content(
        model="gemini-embedding-2",
        contents=content,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION)
    )
    return result.embeddings[0].values