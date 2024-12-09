from transformers import GPT2TokenizerFast
import torch
import Levenshtein, tqdm, collections
import numpy as np
from datetime import datetime

from run_sample import sample_tokens

torch.set_printoptions(threshold=10000)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def check_canonicity_bool(actual_tokens, canonical_tokens):
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


def dist_canon(X_tensor: torch.Tensor) -> np.ndarray:
    X = X_tensor.tolist()
    f = tokenizer.decode if isinstance(X[0], int) else tokenizer.batch_decode
    s = f(X, skip_special_tokens=True)
    K, O = tokenizer(s, add_special_tokens=False)["input_ids"], rmst(X)
    return np.array([Levenshtein.distance(k, o) for k, o in tqdm.tqdm(zip(K, O), total=len(K))])

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
        l_u += len(tokenizer.decode([u])) if move_i else 0
        l_v += len(tokenizer.decode([v])) if move_j else 0
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
            O[i-start_i].append(([tokenizer.decode([V[x]]) for x in range(start_i, i)],
                                 [tokenizer.decode([V_canon[y]]) for y in range(start_j, j)]))
            c = 0
    return O

def check_canonicity_many():
    output_file = "12-2-batch-check-canonicity.txt"
    raw_file = "12-2-batch-raw-check-canonicity.txt"

    device = torch.device("cuda:0")
    tokenizer.pad_token = tokenizer.eos_token

    token_counts = [1, 50, 100, 200, 300, 500]
    step_counts = [100, 100, 100, 100, 100, 100]

    batches = 2 # change to 6
    batch_size = 3 # change to 100
    
    for i in range(batches): 
        token_count = token_counts[i]
        steps = step_counts[i]

        actual_tokens = sample_tokens(batch_size, token_count, steps)
        text = tokenizer.batch_decode(actual_tokens)
        canonical_tokens = tokenizer(
            text, 
            padding=True, 
            truncation=True,
            return_tensors='pt')["input_ids"].to(device)

        canon_count = 0

        for j in range(batch_size):
            actual_sample = actual_tokens[j]
            canonical_sample = canonical_tokens[j]
            text_sample = text[j]
            canon = check_canonicity_bool(actual_sample, canonical_sample)
            with open(raw_file, 'a') as file:
                file.write("For iteration " + str(j) + " of " + str(token_count) + " tokens and " + str(steps) + " steps, here were the results:\n\n")
                file.write("Canonical? " + ("Yes\n" if canon else "No\n"))
                file.write("Actual tokens:\n" + str(actual_sample) + "\n\n")
                file.write("Canonical tokens:\n" + str(canonical_sample) + "\n\n")
                file.write("Text:\n" + text_sample + "\n\n")
                file.write("=================================================================\n\n\n")
            if canon:
                canon_count += 1
        
        percent = (canon_count / batch_size) * 100

        with open(output_file, 'a') as file:
            file.write("For " + str(token_count) + " tokens and " + str(steps) + " steps, percent canonicity was " + str(percent) + "%\n")

def check_edit_distance():
    output_file = "12-8-edit-distance-5.txt"
    raw_file = "12-8-raw-edit-distance-5.txt"

    device = torch.device("cuda:0")
    tokenizer.pad_token = tokenizer.eos_token
    
    token_counts = [500, 700, 1000]
    step_counts = [300, 500, 800]

    batches = 2 # change to 6
    batch_size = 2 # change to 100

    for i in range(batches): 
        token_count = token_counts[i]
        steps = step_counts[i]

        actual_tokens = sample_tokens(batch_size, token_count, steps)
        text = tokenizer.batch_decode(actual_tokens)
        canonical_tokens = tokenizer(
            text, 
            padding=True, 
            truncation=True,
            return_tensors='pt')["input_ids"].to(device)
        distances = dist_canon(actual_tokens)

        with open(raw_file, 'a') as file:
            file.write("RESULTS FOR TOKEN COUNT " + str(token_count) + " AND STEP COUNT " + str(steps) + "\n\n\n" )

        for j in range(batch_size):

            actual_sample = actual_tokens[j]
            canonical_sample = canonical_tokens[j]
            text_sample = text[j]

            uncanons_output = uncanons(actual_sample, canonical_sample)

            with open(raw_file, 'a') as file:
                file.write("=================================================================\n")
                file.write("the edit distance for sequence " + str(j) + " with " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(distances[j]) + "\n\n")
                file.write("Here were the words that were tokenized non-canonically, along with their canonical vs non-canonical tokenizations: \n")
                for position, token_pairs in uncanons_output.items():
                    for non_canon, canon in token_pairs:
                        file.write("Non-canonical: " + str(non_canon) + ", canonical: " + str(canon) + "\n")
                file.write("=================================================================\n\n\n")

        with open(raw_file, 'a') as file:
            file.write("=================================================================")
            file.write("=================================================================\n\n\n\n")

        avg_edit_distance = np.mean(distances)

        with open(output_file, 'a') as file:
            file.write("The average edit distance for " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(avg_edit_distance) + "\n")

def main():
    start_time = datetime.now() 

    check_edit_distance()

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Program completed in {elapsed_time}.")

if __name__ == "__main__":
    main()