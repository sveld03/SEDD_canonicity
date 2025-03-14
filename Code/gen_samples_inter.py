import torch
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast, AutoModelForCausalLM
from run_sample import sample_tokens
from load_model import load_model
from datetime import datetime
import Levenshtein
from utils import compute_perplexity, rhloglikelihood, custom_decode, custom_encode, uncanons, dist_canon, collapse_mask_tokens

start_time = datetime.now() 

# Set up model and tokenizer
device = torch.device("cuda:2")
sedd_model, graph, noise = load_model("louaaron/sedd-medium", device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

auto_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda:2")

# Define parameters
total_samples = 100  # Total samples required per step count
batch_size = 1      # Generate only 1 at a time
token_count = 1024   # Fixed token count
# step_counts = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]
step_counts = [1024]

# Open CSV file for writing and store results incrementally
csv_filename = "intermediate-data-2.csv"
with open(csv_filename, "w") as f:
    f.write("Token Count,Step Count,Sample Index,Step Number,Original Tokens,Decoded Text,Retokenized Tokens,Canonical?,Edit Distance,Original Perplexity,Retokenized Perplexity,Non-Canonical IDs,Canonical IDs,Non-Canonical Strings,Canonical Strings\n")  # CSV Header

# Generate samples
for steps in step_counts:
    print(f"Generating {total_samples} samples for {steps} steps...")

    # Generate 100 samples in batches of 1
    for batch_num in range(total_samples):
        print(f"Sample {batch_num + 1}")
        samples = sample_tokens(1, token_count, steps, intermediates=True)  # Generate 1 sample, with all intermediate steps cached

        results = []
        for step, original_sequence in samples.items():

            # if step == 10:
            #     break # limit the number of steps for testing

            # if step % 100 != 1 and step != steps:
            #     continue

            sample_index = (batch_num+30) * batch_size  # Compute absolute sample index

            original_tokens = original_sequence[0]
            # original_tokens_cpy = original_tokens.copy()

            # if step == 0:
            #     decoded_text = 
                
            # Decode token IDs to text
            decoded_text = custom_decode(tokenizer, original_tokens)

            decoded_text_display = collapse_mask_tokens(decoded_text)

            # Re-tokenize the decoded text
            retokenized_tokens = custom_encode(tokenizer, decoded_text)
            # retokenized_tokens_cpy = retokenized_tokens.copy()

            canon_bool = 1 if (original_tokens == retokenized_tokens) else 0

            edit_distance = dist_canon(original_tokens, retokenized_tokens)[0]

            original_perplexity = compute_perplexity(auto_model, tokenizer, [original_tokens])
            retokenized_perplexity = compute_perplexity(auto_model, tokenizer, [retokenized_tokens[:1024]])  # Limit to 1024 tokens for perplexity calculation

            # Identify non-canonical and canonical tokenizations using `uncanons()`
            # uncanons_output = uncanons(original_tokens, retokenized_tokens, tokenizer)
            # non_canonical_list, canonical_list = [], []

            # for position, token_pairs in uncanons_output.items():
            #     for non_canon, canon in token_pairs:
            #         non_canonical_list.append(" ".join(map(str, non_canon)))  # Convert to readable format
            #         canonical_list.append(" ".join(map(str, canon)))

            dist, segments = uncanons(original_tokens, retokenized_tokens, tokenizer)

            if step % 100 == 1 or step == steps+1:
                non_canonical_list_tokens = []
                canonical_list_tokens = []
                
                non_canonical_list_text = []
                canonical_list_text = []

                # Iterate over the segments returned by the new function
                for seg in segments:
                    # seg["original_tokens"] is a list of token IDs that differ in this segment
                    # seg["canonical_tokens"] is the corresponding list of token IDs in the canonical sequence
                    non_canonical_list_tokens.append(", ".join(map(str, seg["original_tokens"])))
                    canonical_list_tokens.append(", ".join(map(str, seg["canonical_tokens"])))

                # Join mismatches into a single string (or "None" if empty)
                non_canonical_tokens = "; ".join(non_canonical_list_tokens) if non_canonical_list_tokens else "None"
                canonical_tokens = "; ".join(canonical_list_tokens) if canonical_list_tokens else "None"
                
                # Iterate over the segments returned by the new function
                for seg in segments:
                    # seg["original_tokens"] is a list of token IDs that differ in this segment
                    # seg["canonical_tokens"] is the corresponding list of token IDs in the canonical sequence
                    non_canonical_list_text.append(" ".join(map(str, seg["original_text"])))
                    canonical_list_text.append(" ".join(map(str, seg["canonical_text"])))

                # Join mismatches into a single string (or "None" if empty)
                non_canonical_str = "; ".join(non_canonical_list_text) if non_canonical_list_text else "None"
                canonical_str = "; ".join(canonical_list_text) if canonical_list_text else "None"

            else:
                non_canonical_tokens = "N/A"
                canonical_tokens = "N/A"
                non_canonical_str = "N/A"
                canonical_str = "N/A"

            # Append results as a CSV row
            results.append([
                token_count,
                steps,
                sample_index,
                step,
                str(original_tokens),  # Store as string to avoid issues
                decoded_text_display.replace("\n", " "),  # Replace newlines for CSV safety
                str(retokenized_tokens),
                canon_bool,
                edit_distance,
                original_perplexity,
                retokenized_perplexity,
                non_canonical_tokens,
                canonical_tokens,
                non_canonical_str,
                canonical_str
            ])

        # Append data to CSV file in batches
        df = pd.DataFrame(results, columns=["Token Count", "Step Count", "Sample Index", "Step Number", "Original Tokens", "Decoded Text", "Retokenized Tokens", "Canonical?", "Edit Distance", "Original Perplexity", "Retokenized Perplexity", "Non-Canonical Tokens", "Canonical Tokens", "Non-Canonical Strings", "Canonical Strings"])
        df.to_csv(csv_filename, mode='a', header=False, index=False)  # Append mode

print(f"Sample generation complete. Data saved to '{csv_filename}'.")

end_time = datetime.now()
elapsed_time = end_time - start_time
print(f"Program completed in {elapsed_time}.")