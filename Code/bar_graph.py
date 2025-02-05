import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

# Load the CSV file
file_path = "raw_data_indiv.csv"
df = pd.read_csv(file_path)

# Ensure the necessary columns exist
if "Step Count" in df.columns and "Canonical?" in df.columns:
    
    # Sort the dataframe by step count to ensure proper binning
    df = df.sort_values("Step Count")

    # Define bin edges and labels
    #bin_edges = np.arange(1, 1126, 125)  # Bins: [1-125], [126-250], ..., [1001-1125]
    bin_edges = [1, 126, 251, 376, 501, 626, 751, 876, 1001, 1125]
    # bin_labels = [f"{start}-{start+124}" for start in bin_edges[:-1]]

    # Create a new column that categorizes each step count into one of the bins
    # df["Bin"] = pd.cut(df["Step Count"], bins=np.append(bin_edges, 1125), labels=bin_labels, include_lowest=True)
    df["Bin"] = pd.cut(df["Step Count"], bins=bin_edges, include_lowest=True)

    # Compute the percentage of canonical samples per bin
    histogram_data = df.groupby("Bin")["Canonical?"].mean() * 100  # Convert fraction to percentage

    bin_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in histogram_data.index]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    histogram_data.index = bin_labels
    histogram_data.plot(kind="bar", color="royalblue", alpha=0.7)

    # Label the plot
    plt.xlabel("Step Count Ranges")
    plt.ylabel("Percentage of Canonical Samples")
    plt.title("Percentage Canonicity vs. Step Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    
    plt.savefig('canonicity_bar_graph.png', bbox_inches='tight', dpi=300)

else:
    print("Required columns 'Step Count' and 'Canonical' not found in dataset. Please check the CSV structure.")
