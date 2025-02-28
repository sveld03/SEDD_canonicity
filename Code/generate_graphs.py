import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

# Load the CSV file
file_path = "Big_Data/intermediate-data.csv"
df = pd.read_csv(file_path)

# Check that the necessary columns exist
if "Step Number" in df.columns and "Edit Distance" in df.columns:
    # Convert the "Edit Distance" column to numeric, coercing errors to NaN.
    df["Step Number"] = pd.to_numeric(df["Step Number"], errors="coerce")
    df["Edit Distance"] = pd.to_numeric(df["Edit Distance"], errors="coerce")

    # Drop rows where "Edit Distance" could not be converted (if any).
    df = df.dropna(subset=["Step Number", "Edit Distance"])
    
    # Group by step number and compute the mean edit distance for each step
    grouped = df.groupby("Step Number")["Edit Distance"].mean().reset_index()
    grouped = grouped.sort_values("Step Number")

    # Extract step numbers and corresponding average edit distances
    step_numbers = grouped["Step Number"]
    avg_edit_distance = grouped["Edit Distance"]

    # Create a line plot
    plt.figure(figsize=(8, 6))
    plt.plot(step_numbers, avg_edit_distance, color='blue', linewidth=2, label="Avg Edit Distance")

    # Add labels and title
    plt.xlabel("Step Number")
    plt.ylabel("Average Edit Distance")
    plt.title("Average Edit Distance vs. Step Number")
    plt.legend()
    plt.grid(True)

    # Adjust layout and save the plot to a file
    plt.tight_layout()
    plt.savefig('avg_edit_distance_linegraph.png', bbox_inches='tight', dpi=300)
else:
    print("Required columns not found in the dataset. Please check the CSV structure.")