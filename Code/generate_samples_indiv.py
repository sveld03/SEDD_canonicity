import torch
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast, AutoModelForCausalLM
from run_sample import sample_tokens
from load_model import load_model
from datetime import datetime
from utils import uncanons, dist_canon, compute_perplexity, rhloglikelihood, custom_decode
import Levenshtein

start_time = datetime.now() 

# Set up model and tokenizer
device = torch.device("cuda:2")
sedd_model, graph, noise = load_model("louaaron/sedd-medium", device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

auto_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda:2")

# Define parameters
token_count = 1024   # Fixed token count
step_counts = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] # list(range(1, 200, 1))

# Open CSV file for writing and store results incrementally
csv_filename = "indivs-test-2-27.csv"
with open(csv_filename, "w") as f:
    f.write("Token Count,Step Count,Sample Index,Original Tokens,Retokenized Tokens,Original Token Strings,Retokenized Token Strings,Decoded Text,Canonical?,Edit Distance,Original Perplexity,Retokenized Perplexity,Non-Canonicals,Canonicals\n")  # CSV Header

# Generate samples
for index, steps in enumerate(step_counts):
    print(f"Generating sample for {steps} steps...")

    
    sample = sample_tokens(1, token_count, steps)

    results = []
    for idx, original_tokens in enumerate(sample):
        sample_index = index  # Compute absolute sample index

        print(original_tokens)
    
        decoded_text = tokenizer.decode(original_tokens)
        
        # Decode token IDs to text
        # decoded_text_visual = custom_decode(tokenizer, original_tokens[step])

        # Re-tokenize the decoded text
        retokenized_tokens = tokenizer(decoded_text, padding=True, truncation=True)["input_ids"]

        original_token_strings = [tokenizer.decode(token_id) for token_id in original_tokens.tolist()]
        retokenized_token_strings = [tokenizer.decode(token_id) for token_id in retokenized_tokens]

        canon_bool = 1 if (original_tokens.tolist() == retokenized_tokens) else 0

        edit_distance = dist_canon([original_tokens.tolist()], retokenized_tokens)[0]

        original_perplexity = compute_perplexity(auto_model, tokenizer, [original_tokens.tolist()]).item()
        retokenized_perplexity = compute_perplexity(auto_model, tokenizer, [retokenized_tokens]).item()

        # Identify non-canonical and canonical tokenizations using `uncanons()`
        uncanons_output = uncanons(original_tokens, retokenized_tokens)
        non_canonical_list, canonical_list = [], []

        for position, token_pairs in uncanons_output.items():
            for non_canon, canon in token_pairs:
                non_canonical_list.append(" ".join(map(str, non_canon)))  # Convert to readable format
                canonical_list.append(" ".join(map(str, canon)))

        # Join mismatches into a single string (or "None" if empty)
        non_canonical_str = ";; ".join(non_canonical_list) if non_canonical_list else "None"
        canonical_str = ";; ".join(canonical_list) if canonical_list else "None"

        # Append results as a CSV row
        results.append([
            token_count,
            steps,
            sample_index,
            str(original_tokens.tolist()),  # Store as string to avoid issues
            str(retokenized_tokens),
            str(original_token_strings),
            str(retokenized_token_strings),
            decoded_text.replace("\n", " "),  # Replace newlines for CSV safety
            canon_bool,
            edit_distance,
            original_perplexity,
            retokenized_perplexity,
            non_canonical_str,
            canonical_str
        ])

        # Append data to CSV file in batches
        df = pd.DataFrame(results, columns=["Token Count", "Step Count", "Sample Index", "Original Tokens", "Retokenized Tokens", "Original Token Strings", "Retokenized Token Strings", "Decoded Text", "Canonical?", "Edit Distance", "Original Perplexity", "Retokenized Perplexity", "Non-Canonicals", "Canonicals"])
        df.to_csv(csv_filename, mode='a', header=False, index=False)  # Append mode

print(f"Sample generation complete. Data saved to '{csv_filename}'.")

end_time = datetime.now()
elapsed_time = end_time - start_time
print(f"Program completed in {elapsed_time}.")