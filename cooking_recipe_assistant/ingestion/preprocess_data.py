import os   # OS
import json # json
import sys  # System
import re   # Regular Expression
import logging # Logging
from tqdm import tqdm

# Prefect text
from prefect import task
from prefect.logging import get_run_logger

# Splitters
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter

# LLM
#from langchain_community.llms import OpenAI
#from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


# Prompts
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Prefect text
from prefect import task
from prefect.logging import get_run_logger

# Module recipe
from cooking_recipe_assistant.commons.utils import (
    read_document, save_document,
    read_text, save_text
)


SEMANTIC_SEARCH_TOKEN_LIMIT = 400
ELSER_TOKEN_OVERLAP = 50


def parse_recipe_summary(
    answer_str: str
):
    # Extraer los campos necesarios del documento JSON
    formatted_json = None
    try:
        # Cleaning text
        clean_text = re.sub(r'//.*', '', answer_str)
        clean_text = re.sub('\n', '', clean_text)
        json_match = re.search(r'\{.*"summary":.*\}', clean_text, re.DOTALL)
        if json_match:
            json_content = json_match.group(0)
            parsed_json = json.loads(json_content)

            # Formatear el JSON de la forma deseada
            #formatted_json = {"questions": parsed_json["questions"]}
            formatted_json = parsed_json
    except Exception as e:
        #print(f"No se encontró contenido JSON. {text}")
        print(answer_str)
        print("The error is: ", e)
        print("="*100)

    return formatted_json


def parse_extraction(
    extract_text:str
):
    # Pattern
    pattern = r'```json\n(.*?)```'

    # Buscar el JSON en el texto
    match = re.search(pattern, extract_text, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
        json_object = json.loads(json_content)
        #print(json_content)
    return json_object


def build_context(transcript):
    context = ''
    for t in transcript:
        text = t['text']
        ti = t['start']
        context += f'{ti}\t{text}\n\n'
    return context


def generate_extraction(
    raw_data:dict,
    blocks_chain,
    extrac_chain
):
    # Semantic blocks
    request_data = {
        'transcript': build_context(raw_data['transcript'])
    }
    transcript_blocks_str = blocks_chain.invoke(request_data)

    # Data extraction
    request_data = {
        'title': raw_data['title'],
        'transcript': transcript_blocks_str
    }
    extrac_str = extrac_chain.invoke(request_data)

    # Parse answer
    extrac_data = parse_extraction(extrac_str)

    return extrac_data


def generate_extraction(
    raw_data:dict,
    blocks_chain,
    extrac_chain,
    generated_blocks_dir:str,
    generated_extractions_dir:str
):
    logger = get_run_logger()
    doc_id = raw_data['video_id']
    title = raw_data['title']
    transcript = raw_data['transcript']

    # Semantic blocks
    blocks_str_path = os.path.join(generated_blocks_dir, f"{doc_id}.txt")
    if not os.path.exists(blocks_str_path):
        logger.debug(f"Generating blocks for doc_id {doc_id} with path: {blocks_str_path}")
        request_data = {
            'transcript': build_context(transcript)
        }
        transcript_blocks_str = blocks_chain.invoke(request_data)
        save_text(blocks_str_path, transcript_blocks_str)
    else:
        transcript_blocks_str = read_text(blocks_str_path)

    # Data extraction
    extraction_str_path = os.path.join(generated_extractions_dir, f"{doc_id}.txt")
    if not os.path.exists(extraction_str_path):
        # Generated summary
        logger.debug(f"Generating extraction for doc_id {doc_id} with path: {extraction_str_path}")
        request_data = {
            'title': title,
            'transcript': transcript_blocks_str
        }
        extrac_str = extrac_chain.invoke(request_data)
        save_text(extraction_str_path, extrac_str)
    else:
        extrac_str = read_text(extraction_str_path)

    # Parse data extraction
    extrac_data = parse_extraction(extrac_str)
    if extrac_data:
        # Save summary json
        extraction_json_path = os.path.join(generated_extractions_dir, f"{doc_id}.json")
        save_document(extraction_json_path, extrac_data)

    return extrac_data


def generate_document(
    recipe_data: dict,
    extraction_data: dict,
    sent_splitter
):
    # Chunking
    text = ''.join(extraction_data["instructions"])
    chunks = sent_splitter.split_text(text)

    # Extract fields
    document = {
          "doc_id"         : recipe_data["video_id"],
          "meals"          : recipe_data["en_playlist_titles"],
          "title"          : extraction_data["title"],
          "ingredients"    : extraction_data["ingredients"],
          "summary"        : extraction_data["summary"],
          "tips"           : extraction_data["tips"],
          "text"           : text,
          "chunks"         : chunks
    }

    # Returns
    return document


@task(
    name="Building chain", 
    tags=["preprocessing"], 
    #retries=3, 
    #retry_delay_seconds=60
)
def build_chain(
    model_name:str,
    template:str
):
    # Build llm
    if model_name.startswith("llama"):
        llm = ChatOllama(model=model_name)
    elif model_name.startswith("gpt"):
        llm = ChatOpenAI(model_name=model_name)
    else:
        raise Exception(f"Not found model name: {model_name}")
    
    # Build chain
    qa_prompt = ChatPromptTemplate.from_template(
        #input_variables=["title", "text"],
        template=template
    )
    chain = qa_prompt | llm | StrOutputParser()

    return chain


@task(
    name="Building splitter", 
    tags=["preprocessing"], 
    #retries=3, 
    #retry_delay_seconds=60
)
def build_splitter(
    splitter_name:str,
    chunk_size:int,
    chunk_overlap:int
):
    if splitter_name == "TokenText":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    elif splitter_name == "TransformersTokenText":
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif splitter_name == "Recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        Exception(f"Not found splitter name: {splitter_name}")
    
    return splitter
    


@task(
    name="Generating extractions", 
    tags=["preprocessing"], 
    #retries=3, 
    #retry_delay_seconds=60
)
def preprocess_data(
    raw_data_dir: str,
    output_data_dir: str,
    blocks_chain,
    extractions_chain,
    splitter
):
    # Preprocessing dirs
    logger = get_run_logger()
    logger.info(f"[PREPRO-PLAYLIST] raw_data_dir : {raw_data_dir}")
    logger.info(f"[PREPRO-PLAYLIST] dest_data_dir: {output_data_dir}")

    # Setting directories
    propro_document_dir = os.path.join(output_data_dir, 'documents')
    generated_dir = os.path.join(output_data_dir, 'generated')
    generated_blocks_dir = os.path.join(generated_dir, 'blocks')
    generated_extractions_dir = os.path.join(generated_dir, 'extractions')
    os.makedirs(propro_document_dir, exist_ok=True)
    os.makedirs(generated_blocks_dir, exist_ok=True)
    os.makedirs(generated_extractions_dir, exist_ok=True)

    # Iterate files
    success = 0
    errors = 0
    for root, _, files in os.walk(raw_data_dir):
        #logger.info(raw_data_dir, root, len(files))
        for filename in tqdm(files):
        #for filename in files:
            if not filename.endswith(".json"):
                continue

            # Read raw data
            raw_document_path = os.path.join(root, filename)
            raw_data = read_document(raw_document_path)
            doc_id = raw_data['video_id']
            length = raw_data['length']
            # Skip filles
            if (length < 60 * 4.9) or (length > 60 * 15):  # 60 segundos equivalen a 1 minuto
                logger.debug(f"Skip for video_id: {doc_id} out of range")
                continue

            # Generate extractions
            extraction_data = generate_extraction(
                raw_data,
                blocks_chain,
                extractions_chain,
                generated_blocks_dir,
                generated_extractions_dir
            )
            if not extraction_data:
                logger.info(f"Not found extraction data for video_id: {doc_id}")
                errors += 1
                continue

            # Generate docuement
            prepro_doc = generate_document(
                raw_data,
                extraction_data,
                splitter
            )

            # Save document
            prepro_document_path = os.path.join(propro_document_dir, filename)
            save_document(prepro_document_path, prepro_doc)
            success += 1

    # Print stats
    logger.info(f"Stats=> Sucess: {success}, Errors: {errors}")