import os
from tqdm import tqdm


from typing import Any, Dict, Iterable

# Elasticsearh
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.helpers import BulkIndexError
from langchain_text_splitters import TokenTextSplitter
from elasticsearch import helpers

# Model Embedding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

# Prefect
from prefect import task
from prefect.logging import get_run_logger

# Module assitant
from cooking_recipe_assistant.commons.utils import (
    read_document, 
    save_document
)


def generate_actions(
    data: dict,
    index_name:str,
    embedding,
):
    # Build document
    doc_id = data['doc_id']
    title = data["title"]
    summary = data["summary"]
    chunks = data["chunks"]

    # Embedings
    title_vector = embedding.embed_query(title)
    summary_vector = embedding.embed_query(summary)
    chunk_vectors = embedding.embed_documents(chunks)

    documents = [{
        "_index": index_name,
        "_id": f"{doc_id}@{chunk_number:03d}",
        "_source":{
            # Text
            "doc_id"          : doc_id,
            "chunk_number"    : chunk_number,
            "meals"           : data["meals"],
            "title"           : title,
            "ingredients"     : data["ingredients"],
            "summary"         : data["summary"],
            "text"            : chunk,
            "tips"            : data["tips"],
            # Vectors
            "title_vector"    : title_vector,
            "summart_vector"  : summary_vector,
            "text_vector"     : chunk_vector
        }
    } for chunk_number, (chunk, chunk_vector) in enumerate(zip(chunks, chunk_vectors))]
    return documents

@task(
    name="Create Embedding",
    tags=["data", "ingestion", "indexing", "embedding"]
)
def build_embedding(
    model_name:str
):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model


@task(
    name="Create ES connection",
    tags=["data", "ingestion", "indexing", "connection"]
)
def create_connection(
    es_url: str,
    timeout
):
    es = Elasticsearch(
        hosts=[es_url],
        timeout=timeout
    )
    return es


@task(
    name="Manage ES",
    tags=["data", "ingestion", "indexing"]
)
def manage_index(
    es:Elasticsearch, 
    index_name:str, 
    settings:dict, 
    mappings:dict, 
    delete_index:bool=False
):
    logger = get_run_logger()
    if es.indices.exists(index=index_name):
        if delete_index:
            logger.info(f"Index {index_name} exists. Deleting it...")
            es.indices.delete(index=index_name)
            logger.info(f"Index {index_name} deleted!")
        else:
            logger.info(f"Index {index_name} already exists. Skipping creation.")
            return
    es.indices.create(index=index_name, settings=settings, mappings=mappings)
    logger.info(f"Index {index_name} created successfully!")


@task(
    name="Indexing data",
    tags=["data", "ingestion", "indexing"]
)
def index_data(
    es: Elasticsearch,
    index_name: str,
    src_data_dir,
    dest_data_dir,
    embedding,
    thread_count=1, 
    chunk_size=10
) -> None:
    logger = get_run_logger()
    logger.info(f"[INDEX-DATA] index_name   : {index_name}")
    logger.info(f"[INDEX-DATA] src_data_dir : {src_data_dir}")
    logger.info(f"[INDEX-DATA] dest_data_dir: {dest_data_dir}")
    # Load Elastic search
    os.makedirs(dest_data_dir, exist_ok=True)

    logger.info(f"Indexing documents to {index_name}...")
    all_documents = []
    for root, _, files in os.walk(src_data_dir):
        print(src_data_dir, root, len(files))
        for filename in tqdm(files):
            if not filename.endswith(".json"):
                continue

            # Read document
            document_path = os.path.join(root, filename)
            recipe_data = read_document(document_path)
            doc_id = recipe_data["doc_id"]

            # Generate elastic search document
            actions = generate_actions(recipe_data, index_name, embedding)
            all_documents.extend(actions)

    # Calculate the number of batches
    logger.info(f"Indexing documents to {index_name}...")
    success_count = 0
    failed_count = 0
    try:
        for success, _ in helpers.parallel_bulk(
            es,
            all_documents,
            thread_count=thread_count,
            chunk_size=chunk_size,
        ):
            if success:
                success_count += 1
            else:
                failed_count += 1
    except helpers.BulkIndexError as e:
        logger.error("Bulk indexing error:", e)
        for error_detail in e.errors:
            logger.error(error_detail)
    '''
    for action in all_documents:
        try:
            doc_id = action["_id"]
            es.index(index=index_name, body=action)
            logger.info(f"Documento indexado con éxito.")
        except Exception as e:
            logger.error(f"Error al indexar el documento {doc_id}: {e}")
    '''
    

    # Save chunks
    for doc in all_documents:
        filename = f'{doc["_id"]}.json'
        json_doc_path = os.path.join(dest_data_dir, filename)
        save_document(json_doc_path, doc)
