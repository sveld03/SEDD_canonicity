import torch

import os
import logging
from omegaconf import OmegaConf, open_dict

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast
import tqdm, Levenshtein, collections
import numpy as np
import re
# import math, multiprocessing, functools

device = torch.device('cuda:2')
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

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

def collapse_mask_tokens(text):
    """
    Replaces any occurrence of three or more consecutive "[MASK]" tokens (with optional whitespace)
    with a collapsed string: "[MASK] ... [MASK]".
    """
    # This regex matches three or more occurrences of "[MASK]" (optionally surrounded by whitespace)
    pattern = re.compile(r'(?:\s*\[MASK\]\s*){3,}')
    # Replace each match with the collapsed representation.
    collapsed_text = pattern.sub(" [MASK] ... [MASK] ", text)
    # Clean up extra whitespace.
    return re.sub(r'\s+', ' ', collapsed_text).strip()

def show_all_tokens(token_ids):
    """Displays all tokens in the token_ids list, including special tokens."""
    return [tokenizer.decode([t], clean_up_tokenization_spaces=False) for t in token_ids]

# def custom_decode(tokenizer, token_ids):
#     """ Decodes token IDs into text, explicitly displaying '[MASK]' instead of removing it. """
#     token_strings = [tokenizer.decode([t]) for t in token_ids]
#     # tokens = tokenizer.convert_ids_to_tokens(token_ids)
#     tokens = [" [MASK] " if tok_id == 50257 else tok for tok, tok_id in zip(token_strings, token_ids)]
#     return "".join(tokens)  # Join tokens into a readable string

# def custom_encode(tokenizer, text):
#     """Encodes text into token IDs, ensuring '[MASK]' gets correctly mapped back to token ID 50257."""
#     # Replace explicit "[MASK]" with a space, ensuring it's processed correctly
#     text = text.replace(" [MASK] ", tokenizer.pad_token)  
#     tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

#     # Manually replace PAD token ID (if present) back to 50257 for [MASK]
#     tokens = [50257 if tok == tokenizer.pad_token_id else tok for tok in tokens]
    
#     return tokens

def custom_decode(tokenizer, token_ids):
    """Decodes token IDs into text, explicitly displaying '[MASK]' for token ID 50257.
    
    Instead of decoding one token at a time (which can break multi-token Unicode characters),
    we group contiguous non-mask tokens and decode them together.
    """
    output_parts = []
    current_segment = []
    for t in token_ids:
        if t == 50257:  # mask token
            if current_segment:
                # Decode the accumulated non-mask tokens together.
                output_parts.append(tokenizer.decode(current_segment, clean_up_tokenization_spaces=False))
                current_segment = []
            output_parts.append(" [MASK] ")
        else:
            current_segment.append(t)
    if current_segment:
        output_parts.append(tokenizer.decode(current_segment, clean_up_tokenization_spaces=False))
    
    # Combine the parts into a single string.
    return "".join(output_parts)

    #full_text = "".join(output_parts)
    #return collapse_mask_tokens(full_text)

def custom_encode(tokenizer, text):
    """
    Encodes text into token IDs, ensuring that:
      - Explicit "[MASK]" strings are replaced with a placeholder that the tokenizer maps to its pad token,
      - And then that pad token ID is replaced with 50257 (the [MASK] token ID).

    """
    # text = text.replace("[MASK] ... [MASK]", "[MASK]")

    # Replace explicit "[MASK]" with the tokenizer's pad token (keeping spacing intact)
    text = text.replace(" [MASK] ", tokenizer.pad_token)
    
    # Tokenize the modified text without adding special tokens.
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    
    # Convert the pad token IDs back to the [MASK] token ID (50257)
    tokens = [50257 if tok == tokenizer.pad_token_id else tok for tok in tokens]
    
    return tokens

def check_canonicity_one(actual_tokens, canonical_tokens):
    if actual_tokens.numel() != canonical_tokens.numel():
        return False
    if (actual_tokens == canonical_tokens).all():
        return True
    else:
        return False
    
def rmst(X: list) -> list:
    "Returns a new list without special tokens."
    if np.issubdtype(type(X[0]), np.integer) or (len(X[0]) == 0):
        return [x for x in (X.numpy() if isinstance(X, torch.Tensor) else X) if x not in tokenizer.all_special_ids]
    return [[t for t in (x.numpy() if isinstance(x, torch.Tensor) else x) if t not in tokenizer.all_special_ids] for x in X]

def dist_canon(original_tokens: list, retokenized_tokens: list) -> np.ndarray:
    if isinstance(original_tokens, torch.Tensor):
        original_tokens = original_tokens.tolist()
    if isinstance(retokenized_tokens, torch.Tensor):
        retokenized_tokens = retokenized_tokens.tolist()
    return np.array([Levenshtein.distance(original_tokens, retokenized_tokens)])

def process_token_sequence(token_ids: list[int], tokenizer, mask_token_id: int = 50257) -> list[dict[str, any]]:
    """
    Process a token sequence by splitting on mask tokens.
    
    For each non-empty segment (a contiguous block of tokens not equal to mask_token_id):
      - Decode the segment using tokenizer.decode.
      - Re-encode the decoded text (using tokenizer.encode or a custom_encode).
      - Compare the original segment and re-encoded segment.
    
    Returns a list of dictionaries containing the results for each segment.
    """
    segments = []
    current_segment = []
    
    # Split tokens into segments separated by mask tokens.
    for token in token_ids:
        if token == mask_token_id:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            # If current_segment is empty, ignore this mask (or optionally record an empty segment)
        else:
            current_segment.append(token)
    if current_segment:
        segments.append(current_segment)
    
    results = []
    for seg in segments:
        # Decode the segment independently.
        decoded_text = tokenizer.decode(seg, clean_up_tokenization_spaces=False)
        # Re-encode the decoded text.
        # (Optionally, you can use your custom_encode function here if needed.)
        reencoded = tokenizer.encode(decoded_text, add_special_tokens=False)
        # Calculate the edit distance between the original segment and the re-encoded segment.
        distance = dist_canon(seg, reencoded)
        # Determine if the segment is canonical (i.e. the same after re-tokenization).
        is_canonical = (seg == reencoded)
        
        # Store any non-canonical differences if desired (here, we simply store the edit distance)
        results.append({
            "original_segment": seg,
            "decoded_text": decoded_text,
            "reencoded_segment": reencoded,
            "is_canonical": is_canonical,
            "edit_distance": distance
        })
    
    return results

def canon(X: list) -> list:
    f = tokenizer.decode if np.issubdtype(type(X[0]), np.integer) else tokenizer.batch_decode
    s = f(X, skip_special_tokens=False)
    return tokenizer(s, add_special_tokens=False)["input_ids"]

def uncanons(V: list, V_canon: list = None) -> dict:
    if isinstance(V[0], torch.Tensor): V = V.cpu().numpy()
    if V_canon is None: V_canon = canon(V)
    O, c = collections.defaultdict(list), 0
    l_u, l_v = 0, 0
    i, j, start_i, start_j = 0, 0, 0, 0
    move_i, move_j = True, True
    while (i < len(V)) and (j < len(V_canon)):
        u, v = V[i], V_canon[j]
        l_u += len(custom_decode(tokenizer, [u])) if move_i else 0
        l_v += len(custom_decode(tokenizer, [v])) if move_j else 0
        move_i, move_j = False, False
        if l_u >= l_v:
            j += 1
            move_j = True
        if l_v >= l_u:
            i += 1
            move_i = True
        if l_u != l_v:
            if c == 0: start_i, start_j = i-move_i, j-move_j
            c += 1
        elif c > 0:
            O[i-start_i].append(([custom_decode(tokenizer, [V[x]]) for x in range(start_i, i)],
                                 [custom_decode(tokenizer, [V_canon[y]]) for y in range(start_j, j)]))
            c = 0
    return O

def find_unmasked_indices(prev_tokens, current_tokens):
    return [i for i, (p, c) in enumerate(zip(prev_tokens, current_tokens)) if p != c]

def rhloglikelihood(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: list,
                    batch_size: int = 16, prefix: list = [], prefix_cutoff: bool = False,
                    online_memory: bool = True, use_tqdm: bool = False, **kwargs):
    
    if len(prefix) > 0:
        if isinstance(prefix[0], list):
            in_texts = [prefix[i] + texts[i] for i in range(len(texts))]
            if prefix_cutoff:
                cutoff = list(map(len, prefix))
        else:
            in_texts = [prefix + texts[i] for i in range(len(texts))]
            if prefix_cutoff:
                cutoff = np.full(len(in_texts), len(prefix), dtype=np.int32)
        if not prefix_cutoff:
            cutoff = np.zeros(len(in_texts), dtype=np.int32)
    else:
        in_texts = texts
        cutoff = np.ones(len(in_texts), dtype=np.int32)

    # Remove all [MASK] tokens with ID 50257 and ensure sequence isn't empty.
    for text in in_texts:
        filtered = [token for token in text if token != 50257]
        # If filtering removes all tokens, insert the pad token (or another default token)
        if not filtered:
            filtered = [tokenizer.pad_token_id]  # or another fallback token
        text[:] = filtered

    text_len = list(map(len, in_texts))
    d = max(text_len)
    torch.cuda.empty_cache()
    
    _rng = range(0, len(in_texts), batch_size)
    rng = tqdm.tqdm(_rng, desc="Computing log-likelihood") if use_tqdm else _rng
    skip = 1  # Avoid off-by-one errors

    if online_memory:
        n = len(in_texts)
        lls = []
        with torch.no_grad():
            T = torch.full((batch_size, d), tokenizer.pad_token_id, dtype=torch.int32).to(model.device)
            M = torch.zeros((batch_size, d), dtype=torch.int32).to(model.device)

            for batch_idx in rng:
                batch_size_ = min(batch_size, n - batch_idx)
                T_b, M_b = T[:batch_size_], M[:batch_size_]

                for i in range(batch_size_):
                    l = len(in_texts[batch_idx + i])
                    if l > d:
                        print(f"Warning: Text {batch_idx + i} length {l} exceeds d={d}")
                        l = d  # Truncate to prevent indexing errors

                    T_b[i, :l] = torch.tensor(in_texts[batch_idx + i])
                    T_b[i, l:] = tokenizer.pad_token_id
                    M_b[i, :l] = 1
                    M_b[i, l:] = 0  # Ensure padding is ignored

                # print(f"T_b.shape: {T_b.shape}, M_b.shape: {M_b.shape}")  # Debugging

                # Ensure logits are correctly sized
                logits = model(input_ids=T_b, attention_mask=M_b).logits
                logits = logits[:, :-skip, :]  # Prevent off-by-one errors
                log_probs = torch.log_softmax(logits, dim=-1)

                # Check dimensions before indexing
                if log_probs.shape[1] < d - skip:
                    print(f"Warning: log_probs.shape={log_probs.shape} but expected at least {d - skip}")
                    continue  # Skip faulty batch

                log_probs = log_probs[torch.arange(batch_size_).unsqueeze(-1),
                                      torch.arange(d - skip).unsqueeze(0), T_b[:, skip:]]

                log_probs *= M_b[:, skip:]
                lls_ = torch.stack([
                    torch.sum(log_probs[i][cutoff[batch_idx + i] - 1:text_len[batch_idx + i] - 1], dim=-1)
                    for i in range(log_probs.shape[0])
                ], dim=0)

                lls.append(lls_)

        return torch.cat(lls, dim=0).cpu()

    return None  # Should not reach this point

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