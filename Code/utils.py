import torch

import os
import logging
from omegaconf import OmegaConf, open_dict

from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm, numpy as np
# import math, multiprocessing, functools

device = torch.device('cuda:2')


def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def rhloglikelihood(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: list,
                    batch_size: int = 16, prefix: list = [], prefix_cutoff: bool = False,
                    online_memory: bool = True, use_tqdm: bool = False, **kwargs):
    if len(prefix) > 0:
        if isinstance(prefix[0], list):
            in_texts = [prefix[i]+texts[i] for i in range(len(texts))]
            if prefix_cutoff: cutoff = list(map(len, prefix))
        else:
            in_texts = [prefix+texts[i] for i in range(len(texts))]
            if prefix_cutoff: cutoff = np.full(len(in_texts), len(prefix), dtype=np.int32)
        if not prefix_cutoff: cutoff = np.zeros(len(in_texts), dtype=np.int32)
    else:
        in_texts = texts
        cutoff = np.ones(len(in_texts), dtype=np.int32)
    text_len = list(map(len, in_texts))
    d = max(text_len)
    torch.cuda.empty_cache()
    _rng = range(0, len(in_texts), batch_size)
    rng = tqdm.tqdm(_rng, desc="Computing log-likelihood") if use_tqdm else _rng
    skip = int(True)#not is_mamba_tokenizer(tokenizer))

    if online_memory:
        n = len(in_texts)
        lls = []
        with torch.no_grad():
            T = torch.zeros((batch_size, d), dtype=torch.int32).to(device)
            M = torch.zeros((batch_size, d), dtype=torch.int32).to(device)
            for batch_idx in rng:
                batch_size_ = min(batch_size, n - batch_idx)
                T_b, M_b = T[:batch_size_,:], M[:batch_size_,:]
                for i in range(batch_size_):
                    l = len(in_texts[batch_idx+i])
                    T_b[i,:l], T_b[i,l:] = torch.tensor(in_texts[batch_idx+i]), tokenizer.pad_token_id
                    M_b[i,:l], M_b[i,l:] = 1, 0
                # Get logits up to the last token (which would be the suffix) unless skip == 0,
                # in which case returns the whole thing.
                logits = model.forward(input_ids=T_b, attention_mask=M_b).logits[:,:-skip or None,:]
                log_probs = torch.log_softmax(logits, dim=-1)
                log_probs = log_probs[torch.arange(0, batch_size_).unsqueeze(-1),
                                      torch.arange(0, d-skip).unsqueeze(0), T_b[:,skip:]]
                log_probs *= M_b[:,skip:]
                lls_ = torch.cat(tuple(torch.sum(log_probs[i][cutoff[batch_idx+i]-1:text_len[batch_idx+i]-1], dim=-1).reshape(1)
                                       for i in range(log_probs.shape[0])), dim=0)
                lls.append(lls_)

        lls = torch.cat(lls, dim=0)
        return lls.to("cpu")

    T = torch.full((len(texts), d), tokenizer.pad_token_id, dtype=torch.int32).to(model.device)
    M = torch.zeros((len(texts), d), dtype=torch.int32).to(model.device)
    for i in range(len(text_len)):
        T[i,:len(in_texts[i])] = torch.tensor(in_texts[i])
        M[i,:len(in_texts[i])] = 1

    n = T.shape[0]
    lls = []
    with torch.no_grad():
        for batch_idx in rng:
            batch_size_ = min(batch_size, n - batch_idx)
            input_ids_ = T[batch_idx:batch_idx+batch_size_]
            attention_mask_ = M[batch_idx:batch_idx+batch_size_]
            logits = model.forward(input_ids=input_ids_,
                                   attention_mask=attention_mask_).logits[:,:-skip or None,:]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs[torch.arange(0, batch_size_).unsqueeze(-1),
                                  torch.arange(0, d-skip).unsqueeze(0), input_ids_[:,skip:]]
            log_probs *= attention_mask_[:,skip:]
            lls_ = torch.cat(tuple(torch.sum(log_probs[i][cutoff[batch_idx+i]-1:text_len[batch_idx+i]-1], dim=-1).reshape(1)
                                   for i in range(log_probs.shape[0])), dim=0)
            lls.append(lls_)

    lls = torch.cat(lls, dim=0)

    return lls.to("cpu")

import numpy as np
import torch

def compute_perplexity(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: list, batch_size: int = 16, prefix: list = [], prefix_cutoff: bool = False,
                    online_memory: bool = True, use_tqdm: bool = False, **kwargs):
    # Compute log-likelihoods using the rhloglikelihood function
    log_likelihoods = rhloglikelihood(model, tokenizer, texts)
    
    # Compute the total number of words
    text_lengths = list(map(len, texts))
    total_words = np.sum(text_lengths)
    
    # Sum the log-likelihoods
    total_log_likelihood = torch.sum(log_likelihoods).item()
    
    # Compute the average log-likelihood
    avg_log_likelihood = total_log_likelihood / total_words
    
    # Compute the perplexity
    perplexity = np.exp(-avg_log_likelihood)
    
    return perplexity


def batch_rhloglikelihood(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: list,
                          batch_size: int = 16, prefix: list = [], prefix_cutoff: bool = False,
                          online_memory: bool = True, use_tqdm: bool = False, **kwargs):
    if len(prefix) > 0:
        if isinstance(prefix[0], list):
            in_texts = [prefix[i] + texts[i] for i in range(len(texts))]
            if prefix_cutoff: cutoff = list(map(len, prefix))
        else:
            in_texts = [prefix + texts[i] for i in range(len(texts))]
            if prefix_cutoff: cutoff = np.full(len(in_texts), len(prefix), dtype=np.int32)
        if not prefix_cutoff: cutoff = np.zeros(len(in_texts), dtype=np.int32)
    else:
        in_texts = texts
        cutoff = np.ones(len(in_texts), dtype=np.int32)
    text_len = list(map(len, in_texts))
    d = max(text_len)
    torch.cuda.empty_cache()
    _rng = range(0, len(in_texts), batch_size)
    rng = tqdm.tqdm(_rng, desc="Computing log-likelihood") if use_tqdm else _rng
    skip = int(True)  # not is_mamba_tokenizer(tokenizer))

    if online_memory:
        n = len(in_texts)
        lls = []
        with torch.no_grad():
            T = torch.zeros((batch_size, d), dtype=torch.int32).to(device)
            M = torch.zeros((batch_size, d), dtype=torch.int32).to(device)
            for batch_idx in rng:
                batch_size_ = min(batch_size, n - batch_idx)
                T_b, M_b = T[:batch_size_, :], M[:batch_size_, :]
                for i in range(batch_size_):
                    l = len(in_texts[batch_idx + i])
                    T_b[i, :l], T_b[i, l:] = torch.tensor(in_texts[batch_idx + i]), tokenizer.pad_token_id
                    M_b[i, :l], M_b[i, l:] = 1, 0
                logits = model.forward(input_ids=T_b, attention_mask=M_b).logits[:, :-skip or None, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                log_probs = log_probs[torch.arange(0, batch_size_).unsqueeze(-1),
                                      torch.arange(0, d - skip).unsqueeze(0), T_b[:, skip:]]
                log_probs *= M_b[:, skip:]
                lls_ = torch.cat(tuple(torch.sum(log_probs[i][cutoff[batch_idx + i] - 1:text_len[batch_idx + i] - 1], dim=-1).reshape(1)
                                       for i in range(log_probs.shape[0])), dim=0)
                lls.append(lls_)
        lls = torch.cat(lls, dim=0)
        return lls.to("cpu")

    T = torch.full((len(texts), d), tokenizer.pad_token_id, dtype=torch.int32).to(model.device)
    M = torch.zeros((len(texts), d), dtype=torch.int32).to(model.device)
    for i in range(len(text_len)):
        T[i, :len(in_texts[i])] = torch.tensor(in_texts[i])
        M[i, :len(in_texts[i])] = 1
    n = T.shape[0]
    lls = []
    with torch.no_grad():
        for batch_idx in rng:
            batch_size_ = min(batch_size, n - batch_idx)
            input_ids_ = T[batch_idx:batch_idx + batch_size_]
            attention_mask_ = M[batch_idx:batch_idx + batch_size_]
            logits = model.forward(input_ids=input_ids_, attention_mask=attention_mask_).logits[:, :-skip or None, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs[torch.arange(0, batch_size_).unsqueeze(-1),
                                  torch.arange(0, d - skip).unsqueeze(0), input_ids_[:, skip:]]
            log_probs *= attention_mask_[:, skip:]
            lls_ = torch.cat(tuple(torch.sum(log_probs[i][cutoff[batch_idx + i] - 1:text_len[batch_idx + i] - 1], dim=-1).reshape(1)
                                   for i in range(log_probs.shape[0])), dim=0)
            lls.append(lls_)
    lls = torch.cat(lls, dim=0)
    return lls.to("cpu")