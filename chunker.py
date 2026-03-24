import yfinance as yf
import pandas as pd
import torch 
import torch.nn.functional as F
from torchviz import make_dot
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import cv2

import gradio as gr

import lightning as L

from transformers import AutoTokenizer

import time

from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")


import wikipediaapi
import wikipedia


urls = [
    "Napoleon",
    "Napoleonic Wars",
    "First French Empire",
    "World War I",
    "World War II",
    "Calculus",
    "Isaac Newton",
    "Linear algebra",
    "Mathematical statistics",
    "Discrete mathematics"
]

import streamlit as st

def get_texts(urls):
    big_string = ""
    for url in urls:
        wiki = wikipediaapi.Wikipedia(user_agent = "Nikobro82/HistoryApp", language="en")
        page = wiki.page(url)
        text = page.text
        big_string = f"{big_string} {text}"
    return big_string

def get_text_by_search(search):
    wiki = wikipediaapi.Wikipedia(user_agent = "Nikobro82/HistoryApp", language="en")
    results = wikipedia.search(search)
    if len(results) == 0:
        return None, None
    page = wiki.page(results[0])
    print(f"PAGE: {page.fullurl}")
    return page.text, page.fullurl


def chunk_data(keywords, session_state):
    stringText, url = get_text_by_search(keywords)
    if not stringText:
        return None, None
    
    if url in session_state.topics and session_state.current_topic == url:
        print("ALREADY HAVE TOPIC, SAME TOPIC")
        return session_state.topics[url], True
    elif url in session_state.topics and session_state.current_topic != url:
        print("ALREADY HAVE TOPIC, NOT CURRENT TOPIC")
        session_state.current_topic = url
        return session_state.topics[url], True

    stringText = stringText.split(" ")
    chunk_size = 64
    overlap = 32



    chunks = []
    for i in range(0, len(stringText) - chunk_size, chunk_size - overlap):
        sentence = stringText[i : i + chunk_size]
        joined_string = " ".join(sentence)
        chunks.append(joined_string)

    
    dictionary_using = {}

    for chunk in chunks:
        embedding = encoder.encode(sentences=chunk, convert_to_tensor=True)
        dictionary_using[len(dictionary_using)] = {"chunk" : chunk, "embedding" : embedding}

    session_state.topics[url] = dictionary_using
    session_state.current_topic = url

    return dictionary_using, False


def encode(text):
    return encoder.encode(sentences=text, convert_to_tensor=True)
    



