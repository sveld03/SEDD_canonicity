from transformers import GPT2TokenizerFast, AutoModelForCausalLM
import torch
import Levenshtein, tqdm, collections
import numpy as np
from datetime import datetime

from run_sample import sample_tokens
from utils import rhloglikelihood
from load_model import load_model

torch.set_printoptions(threshold=10000)

device = torch.device('cuda:0')
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
            canon = check_canonicity_one(actual_sample, canonical_sample)
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
    output_file = "REAL-12-8-edit-distance-2.txt"
    raw_file = "REAL-12-8-raw-edit-distance-2.txt"

    device = torch.device("cuda:0")
    tokenizer.pad_token = tokenizer.eos_token
    
    token_counts = [100, 250, 400, 500, 750, 900, 1000]
    step_counts = [250, 400, 500, 600, 750, 900, 1000]

    num_counts = 7
    batches_per_count = 20
    batch_size = 5

    for i in range(num_counts): 
        token_count = token_counts[i]
        steps = step_counts[i]

        with open(raw_file, 'a') as file:
            file.write("RESULTS FOR TOKEN COUNT " + str(token_count) + " AND STEP COUNT " + str(steps) + "\n\n\n" )

        all_distances = []

        for j in range(batches_per_count):

            actual_tokens = sample_tokens(batch_size, token_count, steps)
            text = tokenizer.batch_decode(actual_tokens)
            canonical_tokens = tokenizer(
                text, 
                padding=True, 
                truncation=True,
                return_tensors='pt')["input_ids"].to(device)
            distances = dist_canon(actual_tokens)

            for k in range(batch_size):

                actual_sample = actual_tokens[k]
                canonical_sample = canonical_tokens[k]

                uncanons_output = uncanons(actual_sample, canonical_sample)

                with open(raw_file, 'a') as file:
                    file.write("=================================================================\n")
                    file.write("the edit distance for sequence " + str(batch_size*j+k+1) + " with " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(distances[k]) + "\n\n")
                    file.write("Here were the words that were tokenized non-canonically, along with their canonical vs non-canonical tokenizations: \n")
                    for position, token_pairs in uncanons_output.items():
                        for non_canon, canon in token_pairs:
                            file.write("Non-canonical: " + str(non_canon) + ", canonical: " + str(canon) + "\n")
                    file.write("=================================================================\n\n\n")

                all_distances.append(distances[k])

        avg_edit_distance = np.mean(all_distances)

        with open(raw_file, 'a') as file:
            file.write("=================================================================")
            file.write("=================================================================\n\n\n\n")

        with open(output_file, 'a') as file:
            file.write("The average edit distance for " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(avg_edit_distance) + "\n")

def compare_likelihoods():

    output_file = "TEST-12-25-likelihood-1.txt"
    raw_file = "TEST-12-25-raw-likelihood-1.txt"

    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

    device = torch.device("cuda:0")
    tokenizer.pad_token = tokenizer.eos_token
    
    token_counts = [100, 250, 400, 500, 750, 900, 1000]
    step_counts = [250, 400, 500, 600, 750, 900, 1000]

    num_counts = 7
    batches_per_count = 1
    batch_size = 1

    for i in range(num_counts): 
        token_count = token_counts[i]
        steps = step_counts[i]

        with open(raw_file, 'a') as file:
            file.write("RESULTS FOR TOKEN COUNT " + str(token_count) + " AND STEP COUNT " + str(steps) + "\n\n\n" )
        
        all_non_canonical_likelihoods = []
        all_canonical_likelihoods = []

        for j in range(batches_per_count):

            actual_tokens = sample_tokens(batch_size, token_count, steps)
            non_canonical_text = tokenizer.batch_decode(actual_tokens)
            canonical_tokens = tokenizer(
                non_canonical_text, 
                padding=True, 
                truncation=True,
                return_tensors='pt')["input_ids"].to(device)

            for k in range(batch_size):

                non_canonical_likelihood = rhloglikelihood(model, tokenizer, [actual_tokens[k]]).item()
                canonical_likelihood = rhloglikelihood(model, tokenizer, [canonical_tokens[k]]).item()

                uncanons_output = uncanons(actual_tokens[k], canonical_tokens[k])

                with open(raw_file, 'a') as file:
                    file.write("=================================================================\n")
                    file.write("the non-canonical log-likelihood for sequence " + str(batch_size*j+k+1) + " with " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(non_canonical_likelihood) + ", and the canonical log-likeliood is " + str(canonical_likelihood) + "\n\n")
                    for position, token_pairs in uncanons_output.items():
                        for non_canon, canon in token_pairs:
                            file.write("Non-canonical: " + str(non_canon) + ", canonical: " + str(canon) + "\n")
                    file.write("Actual tokens:\n" + str(actual_tokens[k]) + "\n\n")
                    file.write("Canonical tokens:\n" + str(canonical_tokens[k]) + "\n\n")
                    file.write("=================================================================\n\n\n")

                all_non_canonical_likelihoods.append(non_canonical_likelihood)
                all_canonical_likelihoods.append(canonical_likelihood)

        avg_non_canonical_likelihood = np.mean(all_non_canonical_likelihoods)
        avg_canonical_likelihood = np.mean(all_canonical_likelihoods)
        
        with open(raw_file, 'a') as file:
            file.write("=================================================================")
            file.write("=================================================================\n\n\n\n")

        with open(output_file, 'a') as file:
            file.write("the average non-canonical log-likelihood for " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(avg_non_canonical_likelihood) + ", and the average canonical log-likeliood is " + str(avg_canonical_likelihood) + "\n\n")

def do_it_all():
    output_file = "REAL-12-26-doitall-1.txt"
    raw_file = "REAL-12-26-raw-doitall-1.txt"

    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

    device = torch.device("cuda:0")
    tokenizer.pad_token = tokenizer.eos_token
    
    token_counts = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    step_counts = [200, 242, 284, 326, 368, 410, 452, 494, 536, 578, 620, 662, 704, 746, 788, 830, 872, 914, 956, 998]

    num_counts = 20
    batches_per_count = 10
    batch_size = 5

    for i in range(num_counts): 
        token_count = token_counts[i]
        steps = step_counts[i]

        with open(raw_file, 'a') as file:
            file.write("RESULTS FOR TOKEN COUNT " + str(token_count) + " AND STEP COUNT " + str(steps) + "\n\n\n" )

        canon_count = 0
        
        all_distances = []

        all_non_canonical_likelihoods = []
        all_canonical_likelihoods = []

        for j in range(batches_per_count):

            actual_tokens = sample_tokens(batch_size, token_count, steps)
            non_canonical_text = tokenizer.batch_decode(actual_tokens)
            canonical_tokens = tokenizer(
                non_canonical_text, 
                padding=True, 
                truncation=True,
                return_tensors='pt')["input_ids"].to(device)
            
            distances = dist_canon(actual_tokens)

            for k in range(batch_size):

                non_canonical_likelihood = rhloglikelihood(model, tokenizer, [actual_tokens[k]]).item()
                canonical_likelihood = rhloglikelihood(model, tokenizer, [canonical_tokens[k]]).item()

                canon_bool = check_canonicity_one(actual_tokens[k], canonical_tokens[k])

                uncanons_output = uncanons(actual_tokens[k], canonical_tokens[k])

                with open(raw_file, 'a') as file:
                    file.write("=================================================================\n")
                    file.write("Canonical? " + ("Yes\n" if canon_bool else "No\n"))
                    file.write("the edit distance for sequence " + str(batch_size*j+k+1) + " with " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(distances[k]) + "\n\n")
                    file.write("the non-canonical log-likelihood for sequence " + str(batch_size*j+k+1) + " with " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(non_canonical_likelihood) + ", and the canonical log-likeliood is " + str(canonical_likelihood) + "\n\n")
                    for position, token_pairs in uncanons_output.items():
                        for non_canon, canon in token_pairs:
                            file.write("Non-canonical: " + str(non_canon) + ", canonical: " + str(canon) + "\n")
                    file.write("=================================================================\n\n\n")

                if canon_bool:
                    canon_count += 1
                
                all_distances.append(distances[k])

                all_non_canonical_likelihoods.append(non_canonical_likelihood)
                all_canonical_likelihoods.append(canonical_likelihood)

        percent = (canon_count / (batch_size * batches_per_count)) * 100
        avg_edit_distance = np.mean(all_distances)

        avg_non_canonical_likelihood = np.mean(all_non_canonical_likelihoods)
        avg_canonical_likelihood = np.mean(all_canonical_likelihoods)
        
        with open(raw_file, 'a') as file:
            file.write("=================================================================")
            file.write("=================================================================\n\n\n\n")

        with open(output_file, 'a') as file:
            file.write("Here are the results for " + str(token_count) + " tokens and " + str(steps) + " steps: \n\n")
            file.write("Percent canonicity was " + str(percent) + "%\n")
            file.write("The average edit distance is " + str(avg_edit_distance) + "\n")
            file.write("the average non-canonical log-likelihood is " + str(avg_non_canonical_likelihood) + ", and the average canonical log-likeliood is " + str(avg_canonical_likelihood) + "\n")
            file.write("=================================================================\n\n\n")

def find_non_canonicals():
    output_file = "noncanons-1-15-try3.txt"

    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

    device = torch.device("cuda:0")
    tokenizer.pad_token = tokenizer.eos_token

    for i in range(5):
        actual_tokens = sample_tokens(1, 500, 500)
        non_canonical_text = tokenizer.batch_decode(actual_tokens)
        canonical_tokens = tokenizer(
            non_canonical_text, 
            padding=True, 
            truncation=True,
            return_tensors='pt')["input_ids"].to(device)
        
        for k in range(5):
            canon_bool = check_canonicity_one(actual_tokens[k], canonical_tokens[k])
            if not canon_bool:
                uncanons_output = uncanons(actual_tokens[k], canonical_tokens[k])

                with open(output_file, 'a') as file:
                    for position, token_pairs in uncanons_output.items():
                        for non_canon, canon in token_pairs:
                            file.write("Non-canonical: " + str(non_canon) + ", canonical: " + str(canon) + "\n")
                    file.write("=================================================================\n\n")
                    file.write("Text:\n" + str(non_canonical_text[k]) + "\n\n")
                    file.write("=================================================================\n\n\n")
                    file.write("=================================================================\n\n\n")


def main():
    start_time = datetime.now() 

    find_non_canonicals()
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Program completed in {elapsed_time}.")

if __name__ == "__main__":
    main()