import os
import sys
import re
from datetime import datetime
from tqdm.auto import tqdm
import re
from time import time

# Elasticsearch
from elasticsearch import Elasticsearch

# Model
from langchain_huggingface import HuggingFaceEmbeddings

# Prompts
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from .retrievers.es_bm25 import es_bm25_query
from .retrievers.es_hybrid import es_hybrid_query
from .retrievers.es_hybrid_rrf import es_hybrid_rrf_query


def parse_evaluation(
    questions_str: str
):
    # Extraer los campos necesarios del documento JSON
    formatted_json = None
    try:
        # Cleaning text
        clean_text = re.sub(r'//.*', '', questions_str)
        clean_text = re.sub('\n', '', clean_text)
        # Compilar la expresión regular 
        pattern = re.compile(r'"Relevance"\s*:\s*"(.*?)"\s*,?\s*"Explanation"\s*:\s*"(.*?)"(?:\s*[\},])?')
        matches = pattern.findall(clean_text)
        if matches:
            relevance, explanation = matches[0]
            formatted_json = {
                "relevance": relevance,
                "explanation": explanation
            }
        else:
            formatted_json = {
                "relevance": "UNKNOWN",
                "explanation": "Failed to parse evaluation"
            }
    except Exception as e:
        #print(f"No se encontró contenido JSON. {text}")
        print(questions_str)
        #print("The error is: ", e)
        print("="*100)
        formatted_json = {
            "relevance": "UNKNOWN",
            "explanation": "Failed to parse evaluation"
        }

    return formatted_json


def build_recipe_context(search_results, entry_template):
    separator = "\n-----------\n"
    formatted_docs = []
    for doc in search_results:
        doc['meals'] = ', '.join(doc['meals'])
        doc['ingredients'] = ', '.join(doc['ingredients'])
        doc['tips'] = ' '.join([t if t.endswith('.') else t + "." for t in doc['tips']])
        formatted_doc = entry_template.format(**doc)
        #context = context + ENTRY_TEMPLATE.format(**doc) + "\n\n"
        #context = context + doc["text"] + "\n\n"
        formatted_docs.append(formatted_doc)
    #return context.strip()
    return separator.join(formatted_docs).strip()


def build_retriever(es_cnf:dict, entry_template):
    # Create connection
    es_url = es_cnf["url"]
    es_client = Elasticsearch(hosts=[es_url])

    boosting = es_cnf["boosting"]
    index_name = es_cnf["index_name"]
    search_type = es_cnf['type']
    if search_type == 'bm25':
        print(f"search_type: {search_type}")
        retriever = lambda query: build_recipe_context(
            es_bm25_query(es_client, index_name, query, boosting),
            entry_template
        )
    elif search_type == 'hybrid':
        print(f"search_type: {search_type}")
        embedding_model_name = es_cnf["embedding"]["model_name"]
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_field = es_cnf["vector_field"]
        retriever = lambda query: build_recipe_context(
            es_hybrid_query(es_client, index_name, query, embeddings, vector_field, boosting),
            entry_template
        )
    elif search_type == 'hybrid':
        print(f"search_type: {search_type}")
        embedding_model_name = es_cnf["embedding"]["model_name"]
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_field = es_cnf["vector_field"]
        retriever = lambda query: build_recipe_context(
            es_hybrid_rrf_query(es_client, index_name, query, embeddings, vector_field, boosting),
            entry_template
        )
    else:
        raise Exception(f"Not found model name: {search_type}")
    return retriever


def build_llm(
    params:dict
):
    model_name = params['model']
    # Build llm
    if model_name.startswith("llama"):
        print(f"model: {model_name}")
        llm = ChatOllama(**params)
    elif model_name.startswith("gpt"):
        print(f"model: {model_name}")
        llm = ChatOpenAI(**params)
    else:
        raise Exception(f"Not found model name: {model_name}")

    return llm


def build_chain(
    params:dict,
    template:str
):
    # Create prompt
    prompt = ChatPromptTemplate.from_template(
    #prompt = PromptTemplate.from_template(
        template=template
    )
    
    # Build llm
    llm = build_llm(params)
    
    # Build chain
    qa_chain = (
          prompt 
        | llm 
        | StrOutputParser()
    )
    return qa_chain


def build_rag(
    gen_params:dict,
    template:str,
    retriever
):
    # Create prompt
    prompt = ChatPromptTemplate.from_template(
    #prompt = PromptTemplate.from_template(
        template=template
    )
    
    # Build llm
    llm = build_llm(gen_params)
    
    # Build chain
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return qa_chain
