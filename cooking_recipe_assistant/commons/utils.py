import json
import os
import pickle

def read_document(document_path):
    #with open(document_path, 'r') as file:
    with open(document_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_document(document_path, document):
    if len(document) > 0:
        with open(document_path, "w", encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=4)



def read_pickle(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data_path, data):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


def read_text(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    return text


def save_text(text_path, text):
    with open(text_path, 'w') as f:
        f.write(text)