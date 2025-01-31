import torch
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast, AutoModelForCausalLM
from run_sample import sample_tokens
from load_model import load_model
from datetime import datetime
from main import uncanons
import Levenshtein
from utils import compute_perplexity, rhloglikelihood

start_time = datetime.now() 

# Set up model and tokenizer
device = torch.device("cuda:2")
sedd_model, graph, noise = load_model("louaaron/sedd-medium", device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

auto_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda:2")

# Define parameters
total_samples = 100  # Total samples required per step count
batch_size = 5      # Generate only 5 at a time
token_count = 1000   # Fixed token count
step_counts = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]

# Open CSV file for writing and store results incrementally
csv_filename = "raw_data.csv"
with open(csv_filename, "w") as f:
    f.write("Token Count,Step Count,Sample Index,Original Tokens,Decoded Text,Retokenized Tokens,Canonical?,Edit Distance,Original Perplexity,Retokenized Perplexity,Non-Canonicals,Canonicals\n")  # CSV Header

# Generate samples
for steps in step_counts:
    print(f"Generating {total_samples} samples for {steps} steps...")

    # Generate 100 samples in batches of 5
    for batch_num in range(total_samples // batch_size):
        samples = sample_tokens(batch_size, token_count, steps)  # Generate 5 samples

        results = []
        for idx, original_tokens in enumerate(samples):
            sample_index = batch_num * batch_size + idx  # Compute absolute sample index

            # Decode token IDs to text
            decoded_text = tokenizer.decode(original_tokens, skip_special_tokens=True)

            # Re-tokenize the decoded text
            retokenized_tokens = tokenizer(decoded_text, padding=True, truncation=True, add_special_tokens=False)["input_ids"]

            canon_bool = 1 if (original_tokens.tolist() == retokenized_tokens) else 0

            edit_distance = Levenshtein.distance(
                " ".join(map(str, original_tokens.tolist())),
                " ".join(map(str, retokenized_tokens))
            )

            original_perplexity = compute_perplexity(auto_model, tokenizer, [original_tokens]).item()
            retokenized_perplexity = compute_perplexity(auto_model, tokenizer, [retokenized_tokens]).item()

            # Identify non-canonical and canonical tokenizations using `uncanons()`
            uncanons_output = uncanons(original_tokens, retokenized_tokens)
            non_canonical_list, canonical_list = [], []

            for position, token_pairs in uncanons_output.items():
                for non_canon, canon in token_pairs:
                    non_canonical_list.append(" ".join(map(str, non_canon)))  # Convert to readable format
                    canonical_list.append(" ".join(map(str, canon)))

            # Join mismatches into a single string (or "None" if empty)
            non_canonical_str = "; ".join(non_canonical_list) if non_canonical_list else "None"
            canonical_str = "; ".join(canonical_list) if canonical_list else "None"

            # Append results as a CSV row
            results.append([
                token_count,
                steps,
                sample_index,
                str(original_tokens.tolist()),  # Store as string to avoid issues
                decoded_text.replace("\n", " "),  # Replace newlines for CSV safety
                str(retokenized_tokens),
                canon_bool,
                edit_distance,
                original_perplexity,
                retokenized_perplexity,
                non_canonical_str,
                canonical_str
            ])

        # Append data to CSV file in batches
        df = pd.DataFrame(results, columns=["Token Count", "Step Count", "Sample Index", "Original Tokens", "Decoded Text", "Retokenized Tokens", "Canonical?", "Edit Distance", "Original Perplexity", "Retokenized Perplexity", "Non-Canonicals", "Canonicals"])
        df.to_csv(csv_filename, mode='a', header=False, index=False)  # Append mode

print(f"Sample generation complete. Data saved to '{csv_filename}'.")

end_time = datetime.now()
elapsed_time = end_time - start_time
print(f"Program completed in {elapsed_time}.")