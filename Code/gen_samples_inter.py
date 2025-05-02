"""
This script generates and analyzes intermediate steps of the diffusion process, including:
- Token sequence generation with intermediate steps
- MASK token analysis
- Canonicity checking
- Edit distance calculation
- Perplexity computation

The results are saved to a CSV file for further analysis.
"""

import torch
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast, AutoModelForCausalLM
from run_sample import sample_tokens
from load_model import load_model
from datetime import datetime
import Levenshtein
from utils import compute_perplexity, rhloglikelihood, custom_decode, custom_encode, uncanons, dist_canon, collapse_mask_tokens

# -------------------------------
# Configuration
# -------------------------------
# Model and device setup
device = torch.device("cuda:2")
sedd_model, graph, noise = load_model("louaaron/sedd-medium", device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Analysis parameters
total_samples = 100  # Total samples required per step count
batch_size = 1      # Generate only 1 at a time
token_count = 1024   # Fixed token count
step_counts = [1024]  # Step counts to analyze

# Output file
csv_filename = "intermediate-data-2.csv"

# -------------------------------
# Main Execution
# -------------------------------
def main():
    """Main execution function."""
    start_time = datetime.now()
    
    # Initialize CSV file with headers
    with open(csv_filename, "w") as f:
        f.write(
            "Token Count,Step Count,Sample Index,Step Number,"
            "Original Tokens,Decoded Text,Retokenized Tokens,Canonical?,"
            "Edit Distance,Original Perplexity,Retokenized Perplexity,"
            "Non-Canonical IDs,Canonical IDs,Non-Canonical Strings,Canonical Strings\n"
        )
    
    # Load language model for perplexity computation
    auto_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda:2")
    
    # Process each step count
    for steps in step_counts:
        print(f"Generating {total_samples} samples for {steps} steps...")
        
        # Generate samples
        for batch_num in range(total_samples):
            print(f"Sample {batch_num + 1}")
            samples = sample_tokens(1, token_count, steps, intermediates=True)
            
            results = []
            for step, original_sequence in samples.items():
                # Calculate sample index
                sample_index = (batch_num + 30) * batch_size
                original_tokens = original_sequence[0]
                
                # Process tokens
                decoded_text = custom_decode(tokenizer, original_tokens)
                decoded_text_display = collapse_mask_tokens(decoded_text)
                retokenized_tokens = custom_encode(tokenizer, decoded_text)
                
                # Compute metrics
                canon_bool = 1 if (original_tokens == retokenized_tokens) else 0
                edit_distance = dist_canon(original_tokens, retokenized_tokens)[0]
                original_perplexity = compute_perplexity(auto_model, tokenizer, [original_tokens])
                retokenized_perplexity = compute_perplexity(
                    auto_model, 
                    tokenizer, 
                    [retokenized_tokens[:1024]]
                )
                
                # Analyze token differences
                dist, segments = uncanons(original_tokens, retokenized_tokens, tokenizer)
                
                # Process token differences
                if step % 100 == 1 or step == steps + 1:
                    # Extract token IDs
                    non_canonical_list_tokens = []
                    canonical_list_tokens = []
                    for seg in segments:
                        non_canonical_list_tokens.append(
                            ", ".join(map(str, seg["original_tokens"]))
                        )
                        canonical_list_tokens.append(
                            ", ".join(map(str, seg["canonical_tokens"]))
                        )
                    
                    # Extract token strings
                    non_canonical_list_text = []
                    canonical_list_text = []
                    for seg in segments:
                        non_canonical_list_text.append(
                            " ".join(map(str, seg["original_text"]))
                        )
                        canonical_list_text.append(
                            " ".join(map(str, seg["canonical_text"]))
                        )
                    
                    # Format results
                    non_canonical_tokens = "; ".join(non_canonical_list_tokens) or "None"
                    canonical_tokens = "; ".join(canonical_list_tokens) or "None"
                    non_canonical_str = "; ".join(non_canonical_list_text) or "None"
                    canonical_str = "; ".join(canonical_list_text) or "None"
                else:
                    non_canonical_tokens = "N/A"
                    canonical_tokens = "N/A"
                    non_canonical_str = "N/A"
                    canonical_str = "N/A"
                
                # Store results
                results.append([
                    token_count,
                    steps,
                    sample_index,
                    step,
                    str(original_tokens),
                    decoded_text_display.replace("\n", " "),
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
            
            # Save batch results to CSV
            df = pd.DataFrame(
                results,
                columns=[
                    "Token Count", "Step Count", "Sample Index", "Step Number",
                    "Original Tokens", "Decoded Text", "Retokenized Tokens",
                    "Canonical?", "Edit Distance", "Original Perplexity",
                    "Retokenized Perplexity", "Non-Canonical Tokens",
                    "Canonical Tokens", "Non-Canonical Strings", "Canonical Strings"
                ]
            )
            df.to_csv(csv_filename, mode='a', header=False, index=False)
    
    print(f"Sample generation complete. Data saved to '{csv_filename}'.")
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Program completed in {elapsed_time}.")

if __name__ == "__main__":
    main()