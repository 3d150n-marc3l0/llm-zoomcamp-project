from collections import defaultdict
import pandas as pd
import numpy as np
import os
import logging

# files
import json
import pickle

import requests
from bs4 import BeautifulSoup
import csv
import time
from tqdm import tqdm

# Import Youtube
import pytube
from pytube import Playlist
from pytube import Channel
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from pytube.exceptions import PytubeError

# Prefect
from prefect import task
from prefect.logging import get_run_logger

# Translate
from deep_translator import GoogleTranslator


from cooking_recipe_assistant.commons.utils import (
    save_document, save_pickle, read_pickle
)


def translate_text(text):
    traductor = GoogleTranslator(source='es', target='en')
    return traductor.translate(text)


def download_transcript(video_id):
    logger = get_run_logger()
    transcript_doc = {}
    # Get transcript if available
    try:
        #transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['es'])
        if transcript is not None:
            # Transcrip spanish
            es_transcript = transcript.fetch()
            #es_transcript_text = " ".join([entry['text'] for entry in es_transcript])

            # Translation
            en_transcript = transcript.translate('en').fetch()
            #en_transcript_text = " ".join([entry['text'] for entry in en_transcript])

            # Transcription
            #transcript_doc["es_text"] = es_transcript_text
            #transcript_doc["en_text"] = en_transcript_text
            transcript_doc["es_text"] = es_transcript
            transcript_doc["en_text"] = en_transcript
    except Exception as e:
        #print(f"Error retrieving transcript for {video_id}: {e}")
        logger.error(f"Error retrieving transcript for {video_id}: {e}")
        transcript_doc["error"] = "No transcript available"
    return transcript_doc


def download_youtube_video_data(
    video_info: dict
):
    # Retrieve Video
    video_id = video_info['video_id']
    video_url = f'http://youtube.com/watch?v={video_id}'
    video = YouTube(video_url)
    
    # Get transcrtiption
    transcript = download_transcript(video_id)
    if 'error' in transcript:
        raise Exception(f"Not found Transcription for video_id: {video_id}")

    # Extract information
    video_title = video.title
    video_metadata = video.vid_info
    video_description = video.description
    video_keywords = video.keywords
    video_length = video.length
    video_rating = video.rating
    video_views = video.views
    #video_author = video.author
    #video_publish_date = video.publish_date

    # Chanel
    channel_id = video.channel_id
    
    # Translate title
    en_title = translate_text(video.title)
    
    # Build response
    video_data = {
        "title": video_title,
        "en_title": en_title,
        "metadata": video_metadata,
        "description": video_description,
        "keywords": video_keywords,
        "length": video_length,
        "rating": video_rating,
        #"views": video_views,
        #"author": video_author,
        #"publish_date": video_publish_str,
        "transcript": transcript["es_text"],
        "en_transcript": transcript["en_text"],
        "video_id": video_id,
        #"channel_name": channel_name,
        "channel_id": channel_id,
    }
    # Update video data
    video_data['playlist_ids'] = video_info['playlist_ids']
    video_data['playlist_titles'] = video_info['playlist_titles']
    video_data['en_playlist_titles'] = video_info['en_playlist_titles']
    
    # Return data
    return video_data


def download_youtube_videos(
    video_playlist_map: dict, 
    output_dir: str
):
    logger = get_run_logger()
    success_count = 0
    error_count = 0
    # Process each video in the playlist
    for video_id, video_info in tqdm(video_playlist_map.items()):
        # Skip video
        filename = f"{video_id}.json"
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            #print(f"Skip video with video_id: {video_id}. Video was downloaded")
            logger.debug(f"Skip video with video_id: {video_id}. Video was downloaded")
            success_count += 1
            continue
        
        try:
            # Downloading video infos
            logger.info(f"Downloading video {file_path}")
            video_data = download_youtube_video_data(video_info)
            
            # Save video to disk
            save_document(file_path, video_data)
            success_count += 1

            # Wait to avoid Exceptions
            #time.sleep(500 / 1000)
            time.sleep(2)
        except PytubeError as e:
            #print(f"Error retrieving transcript for {video_id}: {e}")
            error_count += 1
            logger.error(f"Error retrieving video info for {video_id}: {e}")
        except Exception as e:
            #print(f"Error retrieving transcript for {video_id}: {e}")
            error_count += 1
            logger.error(f"Error retrieving transcript for {video_id}: {e}")
    # Stats
    logger.info(f"success: {success_count} error: {error_count}")


@task(
    name="generate-dataset", 
    tags=["data"], 
    #retries=3, 
    #retry_delay_seconds=60
)
def download_playlist_info(
    playlist_ids,
    output_dir: str,
    max_videos: int=20,
    update_playlist: bool=True
):

    logger = get_run_logger()
    # Retrive playlist info
    playlist_array = []
    video_playlist_map = {}

    # Retrieve info from Playlits
    for playlist_id in playlist_ids:
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
        #print(f"Downloading playlist {playlist_url}")
        logger.info(f"Downloading playlist {playlist_url}")
        playlist = Playlist(playlist_url)
        #print(f"="*100)
        logger.info(f"="*100)
        playlist_title = playlist.title
        playlist_id = playlist.playlist_id
        playlist_url = playlist.playlist_url
        #playlist_description = playlist.description
        playlist_owner = playlist.owner
        playlist_owner_id = playlist.owner_id
        sidebar_info  = playlist.sidebar_info
        ## Translate
        en_playlist_title = translate_text(playlist_title)
        #print(f"[DOWNLOAD-PLAYLIST-INFO] playlist_id :{playlist.playlist_id}, playlist_tile: {playlist.title}, length: {playlist.length}")
        #print(f"="*100)
        logger.info(f"[DOWNLOAD-PLAYLIST-INFO] playlist_id :{playlist.playlist_id}, playlist_tile: {playlist.title}, length: {playlist.length}")
        logger.info(f"="*100)

        video_count = 0
        for video in tqdm(playlist.videos):
            video_id = video.video_id
            #video_title = video.title
            video_data = {}
            if video_id in video_playlist_map:
                video_data = video_playlist_map[video_id]
            else:
                # video
                video_data['video_id'] = video_id
                #video_data['video_title'] = video_title
                # Playlist info
                video_data['playlist_ids'] = []
                video_data['playlist_titles'] = []
                video_data['en_playlist_titles'] = []
                video_playlist_map[video_id] = video_data

            # Playlits info
            if not playlist_id in video_data['playlist_ids']:
                video_data['playlist_ids'].append(playlist_id)
                video_data['playlist_titles'].append(playlist_title)
                video_data['en_playlist_titles'].append(en_playlist_title)
            video_count += 1
            if (video_count % max_videos) == 0:
              break

        # Add Playlist
        playlist_array.append({
            "playlist_id"       : playlist_id,
            "playlist_title"    : playlist_title,
            "en_playlist_title" : en_playlist_title,
            "playlist_url"      : playlist_url,
            #'playlist_description': playlist_description,
            'video_count'       : video_count #len(playlist.videos)
        })

    # Save info
    if len(video_playlist_map) > 0:
        #print(f"Saving video_playlist_map: {len(video_playlist_map)}")
        file_path = os.path.join(output_dir, 'video_playlist_map.pkl')
        save_pickle(file_path, video_playlist_map)
    if len(playlist_array) > 0:
        #print(f"Saving playlist_array: {len(playlist_array)}")
        file_path = os.path.join(output_dir, 'playlist_info.pkl')
        save_pickle(file_path, playlist_array)

    return video_playlist_map


def generate_youtube_dataset(
    playlist_ids,
    output_dir,
    update_playlist_info=True,
    update_trascripts=True,
    max_videos=25
):
    # Retrive playlist info
    info_raw_data_dir = os.path.join(output_dir, 'info')
    os.makedirs(info_raw_data_dir, exist_ok=True)
    file_path = os.path.join(info_raw_data_dir, 'video_playlist_map.pkl')
    if not update_playlist_info and os.path.exists(file_path):
        video_playlist_map = read_pickle(file_path)
    else:
        video_playlist_map = download_playlist_info(
            playlist_ids, 
            info_raw_data_dir,
            max_videos=max_videos)        
    
    # Retrive transcript
    doc_raw_data_dir = os.path.join(output_dir, 'documents')
    os.makedirs(doc_raw_data_dir, exist_ok=True)
    download_youtube_videos(video_playlist_map, doc_raw_data_dir)