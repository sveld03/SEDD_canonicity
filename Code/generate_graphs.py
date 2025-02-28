import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

# Load the CSV file
file_path = "Big_Data/intermediate-data.csv"
df = pd.read_csv(file_path, low_memory=False)

# Check that the necessary columns exist
required_columns = ["Step Number", "Original Perplexity", "Retokenized Perplexity"]
for col in required_columns:
    if col not in df.columns:
        print(f"Column '{col}' not found in the dataset. Please check the CSV structure.")
        exit()

# Convert the columns to numeric (coercing errors to NaN) and drop rows with NaN values
df["Step Number"] = pd.to_numeric(df["Step Number"], errors="coerce")
df["Original Perplexity"] = pd.to_numeric(df["Original Perplexity"], errors="coerce")
df["Retokenized Perplexity"] = pd.to_numeric(df["Retokenized Perplexity"], errors="coerce")
df.dropna(subset=required_columns, inplace=True)

# Group by Step Number and compute the mean perplexity values for each step
grouped = df.groupby("Step Number")[["Original Perplexity", "Retokenized Perplexity"]].mean().reset_index()
grouped = grouped.sort_values("Step Number")

# Extract the data for plotting
steps = grouped["Step Number"]
avg_original_perplexity = grouped["Original Perplexity"]
avg_retokenized_perplexity = grouped["Retokenized Perplexity"]
avg_perplexity_difference = avg_original_perplexity - avg_retokenized_perplexity

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(steps, avg_perplexity_difference, color="green", linewidth=2, label="Perplexity Difference")

# Add labels and title
plt.xlabel("Step Number")
plt.ylabel("Perplexity Difference (Original - Retokenized)")
plt.title("Average Perplexity Difference vs. Step Number")
plt.legend()
plt.grid(True)

# Adjust layout and save the plot to a file
plt.tight_layout()
plt.savefig("perplexity_difference_vs_steps.png", bbox_inches='tight', dpi=300)