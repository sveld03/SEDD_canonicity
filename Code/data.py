"""
This module provides functionality for loading, preprocessing, and tokenizing various text datasets
including WikiText, PTB, and LAMBADA. It includes custom detokenizers for different datasets and
handles the creation of data loaders for training and evaluation.
"""

import re
import json
import urllib.request
import zipfile
import requests
import numpy as np
import torch
from transformers import GPT2TokenizerFast
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, DistributedSampler
from itertools import chain

# -------------------------------
# Data Loading Utilities
# -------------------------------
def cycle_loader(dataloader, sampler=None):
    """
    Creates an infinite cycle through the dataloader.
    
    Args:
        dataloader (DataLoader): The dataloader to cycle through
        sampler (DistributedSampler, optional): Sampler for distributed training
        
    Yields:
        dict: Batches of data from the dataloader
    """
    while True:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data

# -------------------------------
# Detokenization Functions
# -------------------------------
def wt_detokenizer(string):
    """
    Detokenizes WikiText format text by handling contractions, numbers, punctuation, and brackets.
    
    Args:
        string (str): Tokenized text to detokenize
        
    Returns:
        str: Detokenized text
    """
    # Handle contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    
    # Handle number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    
    # Handle punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    
    # Handle brackets and quotes
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    
    # Handle miscellaneous cases
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    
    return string

def ptb_detokenizer(x):
    """
    Detokenizes Penn Treebank format text.
    
    Args:
        x (str): Tokenized text to detokenize
        
    Returns:
        str: Detokenized text
    """
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    """
    Detokenizes LM1B format text.
    
    Args:
        x (str): Tokenized text to detokenize
        
    Returns:
        str: Detokenized text
    """
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x

def lambada_detokenizer(text):
    """
    Detokenizes LAMBADA format text.
    
    Args:
        text (str): Tokenized text to detokenize
        
    Returns:
        str: Detokenized text
    """
    text = text.replace(""", '"')
    text = text.replace(""", '"')
    return '\n'+text.strip()

# -------------------------------
# Dataset Loading Functions
# -------------------------------
def get_lambada_test_dataset():
    """
    Loads the LAMBADA test dataset from OpenAI's servers.
    
    Returns:
        Dataset: The LAMBADA test dataset
    """
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"
    
    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)
        return data_list
    
    lambada_data = read_jsonl_to_list(url)
    return Dataset.from_list(lambada_data)

def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    """
    Loads and preprocesses a specified dataset.
    
    Args:
        name (str): Name of the dataset to load
        mode (str): Dataset split to load (train/validation/test)
        cache_dir (str, optional): Directory to cache the dataset
        block_size (int): Maximum sequence length
        num_proc (int): Number of processes for parallel processing
        
    Returns:
        Dataset: Preprocessed and tokenized dataset
    """
    # Load the appropriate dataset
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)
    
    # Select the appropriate split
    data = dataset[mode] if name != "lambada" else dataset
    
    # Select the appropriate detokenizer
    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None
    
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    
    def preprocess_and_tokenize(example):
        """Preprocesses and tokenizes a single example."""
        text = example['sentence'] if name == "ptb" else example["text"]
        
        if detokenizer is not None:
            text = detokenizer(text)
        
        tokens = tokenizer(text, return_attention_mask=False)
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    # Process the dataset
    tokenized_dataset = data.map(
        preprocess_and_tokenize, 
        batched=True, 
        num_proc=num_proc, 
        load_from_cache_file=True
    )
    
    # Remove unnecessary columns
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    
    def group_texts(examples):
        """Groups texts into blocks of specified size."""
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        
        return {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    
    # Group texts and format for PyTorch
    chunked_dataset = tokenized_dataset.map(
        group_texts, 
        batched=True, 
        num_proc=num_proc, 
        load_from_cache_file=True
    )
    chunked_dataset = chunked_dataset.with_format('torch')
    
    return chunked_dataset

def get_dataloaders(config, distributed=True):
    """
    Creates training and validation dataloaders.
    
    Args:
        config: Configuration object containing dataset and training parameters
        distributed (bool): Whether to use distributed training
        
    Returns:
        tuple: (train_loader, valid_loader)
        
    Raises:
        ValueError: If batch sizes are not compatible with distributed training
    """
    # Validate batch sizes
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Train Batch Size {config.training.batch_size} is not divisible by "
            f"{config.ngpus} gpus with accumulation {config.training.accum}."
        )
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Eval Batch Size {config.eval.batch_size} is not divisible by "
            f"{config.ngpus} gpus with accumulation {config.training.accum}."
        )
    
    # Load datasets
    train_set = get_dataset(
        config.data.train, 
        "train", 
        cache_dir=config.data.cache_dir, 
        block_size=config.model.length
    )
    valid_set = get_dataset(
        config.data.valid, 
        "validation" if config.data.valid != "text8" else "test", 
        cache_dir=config.data.cache_dir, 
        block_size=config.model.length
    )
    
    # Create samplers for distributed training
    if distributed:
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    
    # Create dataloaders
    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    
    return train_loader, valid_loader

