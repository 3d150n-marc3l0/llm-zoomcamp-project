import os
import json
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

# Elasticsearch
from elasticsearch import Elasticsearch


def compute_rrf(rank, k=60):
    """ Our own implementation of the relevance score """
    return 1 / (k + rank)

def es_hybrid_rrf_query(
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
        "boost": query_boost,
    }

    keyword_query = {
        "multi_match": {
            "query": query,
            "fields": fields,
            "type": "best_fields",
            "boost": vector_boost,
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

    knn_results = es_client.search(
        index=index_name, 
        body=search_query
    )['hits']['hits']
    
    keyword_results = es_client.search(
        index=index_name, 
        body={
            "query": keyword_query, 
            "size": 10
        }
    )['hits']['hits']
    
    rrf_scores = {}
    # Calculate RRF using vector search results
    for rank, hit in enumerate(knn_results):
        doc_id = hit['_id']
        rrf_scores[doc_id] = compute_rrf(rank + 1, k)

    # Adding keyword search result scores
    for rank, hit in enumerate(keyword_results):
        doc_id = hit['_id']
        if doc_id in rrf_scores:
            rrf_scores[doc_id] += compute_rrf(rank + 1, k)
        else:
            rrf_scores[doc_id] = compute_rrf(rank + 1, k)

    # Sort RRF scores in descending order
    reranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top-K documents by the score
    final_results = []
    for doc_id, score in reranked_docs[:5]:
        doc = es_client.get(index=index_name, id=doc_id)
        #final_results.append(doc['_source'])
        d = doc['_source']
        d["id"] = doc["_id"]
        final_results.append(d)
        
    
    return final_results