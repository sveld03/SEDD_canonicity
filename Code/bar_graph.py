"""
This script generates a bar graph (Graphs/canonicity_bar_graph.png) showing the percentage of canonical samples across different
step count ranges.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib backend to non-interactive mode for server environments
plt.switch_backend('Agg')

# -------------------------------
# Configuration Constants
# -------------------------------
INPUT_FILE = "raw_data_indiv.csv"
BIN_EDGES = [1, 126, 251, 376, 501, 626, 751, 876, 1001, 1125]  # Define step count ranges

# -------------------------------
# Main Execution
# -------------------------------
def generate_canonicity_bar_graph():
    """
    Generates a bar graph showing the percentage of canonical samples across different step count ranges.
    The graph is saved as 'canonicity_bar_graph.png'.
    """
    # Load and validate data
    df = pd.read_csv(INPUT_FILE)
    
    if "Step Count" not in df.columns or "Canonical?" not in df.columns:
        print("Required columns 'Step Count' and 'Canonical' not found in dataset. Please check the CSV structure.")
        return
    
    # Sort data by step count for proper binning
    df = df.sort_values("Step Count")
    
    # Categorize step counts into bins
    df["Bin"] = pd.cut(df["Step Count"], bins=BIN_EDGES, include_lowest=True)
    
    # Calculate percentage of canonical samples per bin
    histogram_data = df.groupby("Bin")["Canonical?"].mean() * 100
    
    # Create bin labels for x-axis
    bin_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in histogram_data.index]
    histogram_data.index = bin_labels
    
    # Create and style the plot
    plt.figure(figsize=(10, 6))
    histogram_data.plot(kind="bar", color="royalblue", alpha=0.7)
    
    # Add labels and formatting
    plt.xlabel("Step Count Ranges")
    plt.ylabel("Percentage of Canonical Samples")
    plt.title("Percentage Canonicity vs. Step Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('canonicity_bar_graph.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_canonicity_bar_graph()
