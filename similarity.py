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

import streamlit as st


def get_scores(v1, v2, k):
    numerator = v1 @ v2.T
    similarities = numerator / (torch.norm(v1) * torch.norm(v2, dim=1))
    top_five_k = torch.topk(similarities, k)
    return top_five_k