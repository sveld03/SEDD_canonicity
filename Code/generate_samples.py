import torch
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast, AutoModelForCausalLM
from run_sample import sample_tokens
from load_model import load_model

# Set up model and tokenizer
device = torch.device("cuda:0")
model, graph, noise = load_model("louaaron/sedd-medium", device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

# Define parameters
num_sequences = 100
token_count = 1000 
# step_counts = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
step_counts = [1000]

# List to store results
results = []

# Generate samples
for steps in step_counts:
    print(f"Generating {num_sequences} samples for {steps} steps...")
    
    # Generate 100 sequences of 1000 tokens
    samples = sample_tokens(num_sequences, token_count, steps)
    
    # Process each sample
    for idx, original_tokens in enumerate(samples):
        # Decode token IDs to text
        decoded_text = tokenizer.decode(original_tokens, skip_special_tokens=True)
        
        # Re-tokenize the decoded text
        retokenized_tokens = tokenizer(decoded_text, add_special_tokens=False)["input_ids"]

        # Store results
        results.append({
            "Step Count": steps,
            "Sample Index": idx,
            "Original Tokens": original_tokens.tolist(),  # Convert tensor to list
            "Decoded Text": decoded_text,
            "Retokenized Tokens": retokenized_tokens
        })

# Convert to DataFrame
df = pd.DataFrame(results)

# Save to CSV file
df.to_csv("generated_samples.csv", index=False)

print("Sample generation complete. Data saved to 'generated_samples.csv'.")
