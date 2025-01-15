import re
import pandas as pd

# Input and output files
input_file = "Real_Data/REAL-12-24-raw-likelihood-1.txt"
output_file = "likelihoods.csv"

# Regular expression to extract data from each line
pattern = r"the non-canonical log-likelihood for sequence \d+ with (\d+) tokens and (\d+) steps is ([^,]+), and the canonical log-likeliood is ([^\.]+)"

# Initialize an empty list to store rows
data = []

# Read the input file
with open(input_file, "r") as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            # Extract token count, step count, and log-likelihoods
            tokens = int(match.group(1))
            steps = int(match.group(2))
            actual_log_likelihood = float(match.group(3))
            canonical_log_likelihood = float(match.group(4))
            # Append as a row to the data list
            data.append([tokens, steps, actual_log_likelihood, canonical_log_likelihood])

# Create a DataFrame
columns = ["Token Count", "Step Count", "Non-Canonical Log-Likelihood", "Canonical Log-Likelihood"]
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame as a CSV file
df.to_csv(output_file, index=False)

print(f"Data successfully extracted and saved to {output_file}")
