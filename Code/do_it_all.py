from transformers import GPT2TokenizerFast, AutoModelForCausalLM
import torch
import Levenshtein, tqdm, collections
import numpy as np
from datetime import datetime

from run_sample import sample_tokens
from utils import rhloglikelihood, batch_rhloglikelihood
from load_model import load_model

# from joblib import Parallel, delayed

torch.set_printoptions(threshold=10000)

device = torch.device('cuda:2')
# model, graph, noise = load_model("louaaron/sedd-medium", device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

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


def dist_canon(X_tensor: torch.Tensor) -> np.ndarray:
    X = X_tensor.tolist()
    f = tokenizer.decode if isinstance(X[0], int) else tokenizer.batch_decode
    s = f(X, skip_special_tokens=True)
    K, O = tokenizer(s, add_special_tokens=False)["input_ids"], rmst(X)
    return np.array([Levenshtein.distance(k, o) for k, o in zip(K, O)])

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

def do_it_all():
    output_file = "TEST-1-30-doitall-optimized.txt"
    raw_file = "TEST-1-30-raw-doitall-optimized.txt"

    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda:2")

    token_count = 1000
    step_counts = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    num_counts = 4
    batches_per_count = 2
    batch_size = 3

    for i in range(num_counts):
        steps = step_counts[i]

        batch_results = []
        all_distances = []
        all_non_canonical_likelihoods = []
        all_canonical_likelihoods = []
        canon_count = 0

        for j in range(batches_per_count):
            # Generate actual token sequences
            actual_tokens = sample_tokens(batch_size, token_count, steps)

            # Decode tokens only once
            non_canonical_texts = tokenizer.batch_decode(actual_tokens)

            # Compute canonical tokens only once
            canonical_tokens = tokenizer(
                non_canonical_texts, padding=True, truncation=True, return_tensors='pt'
            )["input_ids"].to(device)

            # Compute edit distances in batch
            distances = dist_canon(actual_tokens)

            # Compute log-likelihoods in batch
            non_canonical_likelihoods = batch_rhloglikelihood(model, tokenizer, actual_tokens)
            canonical_likelihoods = batch_rhloglikelihood(model, tokenizer, canonical_tokens)

            # Check canonicity in batch
            canon_booleans = [check_canonicity_one(actual_tokens[k], canonical_tokens[k]) for k in range(batch_size)]

            # Compute uncanonicalized token mappings in batch (parallelized)
            uncanons_outputs = [uncanons(actual_tokens[k], canonical_tokens[k]) for k in range(batch_size)]

            # Process results
            for k in range(batch_size):
                if canon_booleans[k]:
                    canon_count += 1

                all_distances.append(distances[k])
                all_non_canonical_likelihoods.append(non_canonical_likelihoods[k])
                all_canonical_likelihoods.append(canonical_likelihoods[k])

                # Store results in a list instead of writing to disk multiple times
                batch_results.append(
                    f"=================================================================\n"
                    f"Canonical? {'Yes' if canon_booleans[k] else 'No'}\n"
                    f"Edit Distance for Seq {batch_size * j + k + 1}: {distances[k]}\n"
                    f"Non-Canonical Log-Likelihood: {non_canonical_likelihoods[k]}, "
                    f"Canonical Log-Likelihood: {canonical_likelihoods[k]}\n"
                )
                for position, token_pairs in uncanons_outputs[k].items():
                    for non_canon, canon in token_pairs:
                        batch_results.append(f"Non-canonical: {non_canon}, Canonical: {canon}\n")

                batch_results.append("=================================================================\n\n")

        # Compute final statistics efficiently using NumPy
        percent = (canon_count / (batch_size * batches_per_count)) * 100
        avg_edit_distance = np.mean(all_distances)
        avg_non_canonical_likelihood = np.mean(all_non_canonical_likelihoods)
        avg_canonical_likelihood = np.mean(all_canonical_likelihoods)

        # Write batch results to file in one go
        with open(raw_file, 'a') as file:
            file.writelines(batch_results)

        # Write summary statistics
        with open(output_file, 'a') as file:
            file.write(
                f"Token Count: {token_count}, Steps: {steps}\n"
                f"Percent Canonicity: {percent}%\n"
                f"Avg Edit Distance: {avg_edit_distance}\n"
                f"Avg Non-Canonical Log-Likelihood: {avg_non_canonical_likelihood}\n"
                f"Avg Canonical Log-Likelihood: {avg_canonical_likelihood}\n"
                f"=================================================================\n\n"
            )

def main():
    start_time = datetime.now() 

    do_it_all()
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Program completed in {elapsed_time}.")

if __name__ == "__main__":
    main()