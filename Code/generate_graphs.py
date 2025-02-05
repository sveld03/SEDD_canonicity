import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

plt.switch_backend('Agg')

# Load the CSV file
file_path = "raw_data_indiv.csv"
df = pd.read_csv(file_path)

# Ensure the relevant columns exist
if "Log Step Count" in df.columns and "Original Log-Likelihood" in df.columns:
    # Extract relevant data
    step_counts = df["Log Step Count"]
    log_likelihoods = df["Original Log-Likelihood"]

    # Perform linear regression to get the line of best fit
    slope, intercept, r_value, p_value, std_err = linregress(step_counts, log_likelihoods)
    line_fit = slope * step_counts + intercept

    # Create scatterplot
    plt.figure(figsize=(8, 6))
    plt.scatter(step_counts, log_likelihoods, label="Data points", color='blue', alpha=0.6)
    plt.plot(step_counts, line_fit, color='red', label=f"Best fit: y={slope:.2f}x + {intercept:.2f}")

    # Add labels and title
    plt.xlabel("Log Step Count")
    plt.ylabel("Original Log-Likelihood")
    plt.title("Original Log-Likelihood vs. Log Step Count")
    plt.legend()
    plt.grid(True)

    # Display statistical values on the graph
    stats_text = (f"Correlation coefficient (r): {r_value:.4f}\n"
                  f"p-value: {p_value:.4e}\n"
                  f"Standard error: {std_err:.4f}")
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('original_log_log_likelihoods_scatterplot.png', bbox_inches='tight', dpi=300)

else:
    print("Required columns not found in the dataset. Please check the CSV structure.")