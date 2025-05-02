# This script was used to generate Graphs/stepwise_perplexity_canonical.png

"""
Script for generating a scatter plot showing the relationship between step number and perplexity,
split by canonicity. The plot includes linear regression lines and statistical annotations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Set matplotlib backend to non-interactive mode for server environments
plt.switch_backend('Agg')

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
# Load the CSV file
file_path = "intermediate-data.csv"
df = pd.read_csv(file_path)

# Ensure the necessary columns exist
if "Step Number" in df.columns and "Canonical?" in df.columns and "Original Perplexity" in df.columns:
    
    # Convert the necessary columns to numeric
    df["Step Number"] = pd.to_numeric(df["Step Number"], errors="coerce")
    df["Canonical?"] = pd.to_numeric(df["Canonical?"], errors="coerce")
    df["Original Perplexity"] = pd.to_numeric(df["Original Perplexity"], errors="coerce")
    
    # Filter for steps 0 to 1025
    df = df[(df["Step Number"] >= 0) & (df["Step Number"] <= 1025)]
    
    # Split data based on canonicity at that exact step
    canonical_df = df[df["Canonical?"] == 1]
    non_canonical_df = df[df["Canonical?"] == 0]
    
    # -------------------------------
    # Plot Generation
    # -------------------------------
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for non-canonical points (red)
    plt.scatter(non_canonical_df["Step Number"], non_canonical_df["Original Perplexity"], 
                color="red", label="Non-Canonical", alpha=0.6)
    
    # Scatter plot for canonical points (blue)
    plt.scatter(canonical_df["Step Number"], canonical_df["Original Perplexity"], 
                color="blue", label="Canonical", alpha=0.6)
    
    # -------------------------------
    # Linear Regression Analysis
    # -------------------------------
    # Perform linear regression for non-canonical points if any
    if not non_canonical_df.empty:
        slope_nc, intercept_nc, r_nc, p_nc, std_err_nc = linregress(
            non_canonical_df["Step Number"], non_canonical_df["Original Perplexity"])
        best_fit_nc = slope_nc * non_canonical_df["Step Number"] + intercept_nc
        plt.plot(non_canonical_df["Step Number"], best_fit_nc, color="darkred", linestyle="--", 
                 label=f"Non-Canonical Fit: y={slope_nc:.2f}x+{intercept_nc:.2f}")
    
    # Perform linear regression for canonical points if any
    if not canonical_df.empty:
        slope_c, intercept_c, r_c, p_c, std_err_c = linregress(
            canonical_df["Step Number"], canonical_df["Original Perplexity"])
        best_fit_c = slope_c * canonical_df["Step Number"] + intercept_c
        plt.plot(canonical_df["Step Number"], best_fit_c, color="darkblue", linestyle="--", 
                 label=f"Canonical Fit: y={slope_c:.2f}x+{intercept_c:.2f}")
    
    # -------------------------------
    # Statistical Annotations
    # -------------------------------
    # Compute location for statistical text
    text_x = df["Step Number"].min() + 0.05 * (df["Step Number"].max() - df["Step Number"].min())
    text_y = df["Original Perplexity"].min() + 0.85 * (df["Original Perplexity"].max() - df["Original Perplexity"].min())
    
    # Format statistical information
    text_str = ""
    if not non_canonical_df.empty:
        text_str += (f"Non-Canonical: r = {r_nc:.4f}, p = {p_nc:.4e}, SE = {std_err_nc:.4f}\n")
    if not canonical_df.empty:
        text_str += (f"Canonical: r = {r_c:.4f}, p = {p_c:.4e}, SE = {std_err_c:.4f}")
    plt.text(text_x, text_y, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # -------------------------------
    # Plot Formatting
    # -------------------------------
    plt.xlabel("Step Number")
    plt.ylabel("Original Perplexity")
    plt.title("Original Perplexity vs. Step Number\n(Color-coded by Canonicity at that Step)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("stepwise_perplexity_canonical.png", bbox_inches='tight', dpi=300)
else:
    print("Required columns not found in the dataset. Please check the CSV structure.")