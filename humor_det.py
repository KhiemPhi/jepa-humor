import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import random
#from tensorrt_llm import SamplingParams
#from tensorrt_llm._torch import LLM
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim

# BERT benchmark
checkpoint = 'distilbert/distilbert-base-uncased'
model_bert = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer_bert = AutoTokenizer.from_pretrained(checkpoint)

masked_ds_humor = load_dataset("allenai/swag", "regular")
