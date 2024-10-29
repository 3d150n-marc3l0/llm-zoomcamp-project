import pandas as pd
import yaml
import os
#from sklearn.feature_extraction.text import TfidfVectorizer

# Prefect
from prefect import flow, task
from prefect.logging import get_run_logger
from prefect.task_runners import SequentialTaskRunner
#from prefect.deployments import Deployment
#from prefect.infrastructure import Process

# tasks
#from preprocess_data import run_data_prep
#from split import split_data
#from hpo import run_optimization, hpo_xgboost, mlflow_environment, HPO_EXPERIMENT_NAME
#from register_model import run_register_model
from cooking_recipe_assistant.commons.utils import (
    read_document
)

from .generate_dataset import generate_youtube_dataset
#from .preprocess_data import preprocess_video_dataset
#rom .generate_summaries import generate_summaries, build_chain
from .preprocess_data import (
    build_splitter,
    build_chain,
    preprocess_data
)
from .es_indexing import (
    create_connection,
    build_embedding,
    manage_index,
    index_data
)


@task(
    name="Read-configuration", 
    tags=["data"], 
    #retries=3, 
    #retry_delay_seconds=60
)
def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


@flow(
    #name="Sentiment-Analysis-Flow",
    description="A flow to run the pipeline for the cooking recipe ingestion",
    task_runner=SequentialTaskRunner()
)
def cooking_recipe_ingestion_flow(config_path: str):

    # ========================================
    # Read config file
    # ========================================
    print(config_path)
    logger = get_run_logger()
    # Read config
    config = read_config(config_path)

    # ========================================
    # Generate dataset
    # ========================================
    #generating_data(config['GENERATE_DATASET'])
    generate_youtube_dataset(
        playlist_ids=config['GENERATE_DATASET']['PLAYLIST_IDS'],
        output_dir=config['GENERATE_DATASET']['OUTPUT_DIR'],
        update_playlist_info=config['GENERATE_DATASET']['UPDATE_PLAYLIST_INFO'],
        update_trascripts=config['GENERATE_DATASET']['UPDATE_TRANSCRIPTS'],
        max_videos=config['GENERATE_DATASET']['MAX_VIDEOS'],
    )
    
    # ========================================
    # Preprocessing
    # ========================================

    prepro_conf = config['PREPROCESSING_DATA']
    logger.info(prepro_conf)

    # Create data clean chain
    blocks_chain = build_chain(
        model_name=prepro_conf['BLOCKS_CHAIN']['MODEL_NAME'],
        template=prepro_conf['BLOCKS_CHAIN']['TEMPLATE_PATH']
    )

    # Create data extraction chain
    extractions_chain = build_chain(
        model_name=prepro_conf['EXTRACTIONS_CHAIN']['MODEL_NAME'],
        template=prepro_conf['EXTRACTIONS_CHAIN']['TEMPLATE_PATH']
    )

    # Create Splitter
    splitter = build_splitter(
        splitter_name=prepro_conf['SPLITTER']['SPLITTER_NAME'],
        chunk_size=prepro_conf['SPLITTER']['CHUNK_SIZE'],
        chunk_overlap=prepro_conf['SPLITTER']['CHUNK_OVERLAP']
    )

    # Preprocess data
    preprocess_data(
        raw_data_dir=prepro_conf['RAW_DATA_DIR'],
        output_data_dir=prepro_conf['PREPROCESSED_DATA_DIR'],
        blocks_chain=blocks_chain,
        extractions_chain=extractions_chain,
        splitter=splitter
    )
    
    # ========================================
    # Indexing data
    # ========================================
    prepro_conf = config['INDEXING_DATA']
    print(prepro_conf)
    print(f"DELETE_INDEX=> value: {prepro_conf['DELETE_INDEX']}, type: {type(prepro_conf['DELETE_INDEX'])}")

    # Create ES connetion
    es_client = create_connection(
        prepro_conf["ES_URL"],
        prepro_conf["TIMEOUT"]
    )

    # Create embedding
    embedding = build_embedding(model_name=prepro_conf["EMBEDDING"]["MODEL_NAME"])
    
    # Manage index
    logger.info(f"DELETE_INDEX=> value: {prepro_conf['DELETE_INDEX']}, type: {type(prepro_conf['DELETE_INDEX'])}")
    index_config = read_document(prepro_conf['INDEX_CONFIG_PATH'])
    manage_index(
        es=es_client,
        index_name=prepro_conf['INDEX_NAME'],
        settings=index_config['settings'],
        mappings=index_config['mappings'],
        delete_index=prepro_conf['DELETE_INDEX']
    )

    # Index
    index_data(
        es=es_client,
        index_name=prepro_conf['INDEX_NAME'],
        src_data_dir=prepro_conf["PREPROCESSED_DOCUMENTS_DIR"],
        dest_data_dir=prepro_conf["INDEXED_DOCUMENTS_DIR"],
        embedding=embedding
    )


if __name__ == '__main__':
    #run_workflow()
    cooking_recipe_ingestion_flow.serve(
        name="cooking-recipe-ingestion-deployment",
        parameters={
            "config_path": "config.yaml"
        }
    )
    