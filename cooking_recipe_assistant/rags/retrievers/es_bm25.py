import os
import json
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

# Elasticsearch
from elasticsearch import Elasticsearch

# Local URL
ES_URL = "http://localhost:9200"

# Index-name
#INDEX_NAME = "cooking-recipes"

# FIeld Names
#FIELD_NAMES = ["meals", "title", "ingredients", "summary", "text", "tips"]

# Boosts
FIELD_BOOOST = {
    'playlist_titles': 3.2332016267009815,
    'title': 1.7979863533110092,
    'ingredients': 3.333797763386082,
    'summary': 4.239107430823736,
    'text': 4.838094409752021
}

def es_bm25_query(
    es_client: Elasticsearch, 
    index_name: str,
    query: str,
    boosts: dict = None
):
    fields = ["meals", "title", "ingredients", "summary", "text", "tips"]
    if boosts:
        #print("Boosting")
        fields = [f"{field}^{boost}" if boost > 0 else field for field, boost in boosts.items()]
        
    search_query = {
        "size": 5,
        "query": {
            "multi_match": {
                "query": query,
                "fields": fields,
                #"type": "best_fields"
            }
        },
        "_source": {
            "excludes": [ "*_vector" ]
        }
    }
    #print(search_query)
    response = es_client.search(
        index=index_name, 
        body=search_query
    )
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        d = hit['_source']
        d["id"] = hit["_id"]
        result_docs.append(d)
    
    return result_docs
