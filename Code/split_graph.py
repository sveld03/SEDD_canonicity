import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

plt.switch_backend('Agg')

# Load the CSV file
file_path = "Real_Data/raw_data_indiv.csv"
df = pd.read_csv(file_path)

# Ensure the necessary columns exist
if "Step Count" in df.columns and "Canonical?" in df.columns:
    
    # Separate canonical and non-canonical data
    canonical_df = df[df["Canonical?"] == 1]
    non_canonical_df = df[df["Canonical?"] == 0]

    # Scatter plot points
    plt.figure(figsize=(10, 6))
    plt.scatter(non_canonical_df["Step Count"], non_canonical_df["Original Log-Likelihood"], 
                color="red", label="Non-Canonical", alpha=0.6)
    plt.scatter(canonical_df["Step Count"], canonical_df["Original Log-Likelihood"], 
                color="green", label="Canonical", alpha=0.6)

    # Perform linear regression for non-canonical
    slope_nc, intercept_nc, r_nc, p_nc, std_err_nc = linregress(
        non_canonical_df["Step Count"], non_canonical_df["Original Log-Likelihood"])
    best_fit_nc = slope_nc * non_canonical_df["Step Count"] + intercept_nc
    plt.plot(non_canonical_df["Step Count"], best_fit_nc, color="darkred", linestyle="--", 
             label=f"Non-Canonical Fit: y={slope_nc:.2f}x+{intercept_nc:.2f}")

    # Perform linear regression for canonical
    slope_c, intercept_c, r_c, p_c, std_err_c = linregress(
        canonical_df["Step Count"], canonical_df["Original Log-Likelihood"])
    best_fit_c = slope_c * canonical_df["Step Count"] + intercept_c
    plt.plot(canonical_df["Step Count"], best_fit_c, color="darkgreen", linestyle="--", 
             label=f"Canonical Fit: y={slope_c:.2f}x+{intercept_c:.2f}")

    # Add statistical annotations
    text_x = min(df["Step Count"]) + (max(df["Step Count"]) - min(df["Step Count"])) * 0.05
    text_y = min(df["Original Log-Likelihood"]) + (max(df["Original Log-Likelihood"]) - min(df["Original Log-Likelihood"])) * 0.85

    text_str = (f"Non-Canonical:\n"
                f"r = {r_nc:.4f}, p = {p_nc:.4e}, SE = {std_err_nc:.4f}\n\n"
                f"Canonical:\n"
                f"r = {r_c:.4f}, p = {p_c:.4e}, SE = {std_err_c:.4f}")

    plt.text(text_x, text_y, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # Labels and title
    plt.xlabel("Step Count")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood vs. Step Count (Canonical vs. Non-Canonical)")
    plt.legend()
    plt.grid(True)

    # Adjust layout and display the plot
    plt.tight_layout()
    
    plt.savefig('split_canonicity.png', bbox_inches='tight', dpi=300)

else:
    print("Required columns 'Step Count', 'Edit Distance', and 'canonical?' not found in dataset. Please check the CSV structure.")