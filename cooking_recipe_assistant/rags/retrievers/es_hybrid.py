import os
import json
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

# Elasticsearch
from elasticsearch import Elasticsearch


def es_hybrid_query(
    es_client: Elasticsearch,
    index_name: str,
    query: str,
    embeddings,
    field: str = "text_vector", 
    boosts: dict = None,
    k:int=10,
    n:int=100
):

    vector_boost = 0.5
    query_boost = 1.0 - vector_boost
    fields = ["meals", "title", "ingredients", "summary", "text", "tips"]
    if boosts:
        fields = [f"{field}^{boost}" if boost > 0 else field for field, boost in boosts.items() 
                  if field in fields]
        if 'vector_boost' in boosts:
            vector_boost = boosts['vector_boost']
            query_boost = 1.0 - vector_boost
    
    vector = embeddings.embed_query(query)
    knn_query = {
        "field": field,
        "query_vector": vector,
        "k": k,
        "num_candidates": n,
        "boost": vector_boost,
    }

    keyword_query = {
        "multi_match": {
            "query": query,
            "fields": fields,
            "type": "best_fields",
            "boost": query_boost,
        }
    }

    search_query = {
        "knn": knn_query,
        "query": keyword_query,
        "size": 5,
        "_source": {
            "excludes": [ "*_vector" ]
        }
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    for hit in es_results['hits']['hits']:
        d = hit['_source']
        d["id"] = hit["_id"]
        result_docs.append(d)
        #result_docs.append(hit['_source'])

    return result_docs

