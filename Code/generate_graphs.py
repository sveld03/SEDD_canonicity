"""
This script generates various visualization graphs for analyzing the diffusion process,
including edit distances, transition frequencies, canonicity metrics, and perplexity measurements.
The graphs are saved in the current directory with 'inter_' prefix.
"""

# Standard library imports
from collections import Counter

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt

# Set matplotlib backend to non-interactive mode for server environments
plt.switch_backend('Agg')

# -------------------------------
# Configuration Constants
# -------------------------------
INPUT_FILE = "intermediate-data.csv"
NUMERIC_COLUMNS = [
    "Step Number",
    "Edit Distance",
    "Sample Index",
    "Canonical?",
    "Original Perplexity",
    "Retokenized Perplexity"
]

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
def load_and_preprocess_data():
    """
    Loads the CSV data and converts specified columns to numeric format.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe with numeric columns properly formatted
    """
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # Convert specified columns to numeric type
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def prepare_plot_dataframes(df):
    """
    Prepares separate dataframes for different types of plots.
    
    Args:
        df (pd.DataFrame): The main dataframe containing all data
        
    Returns:
        tuple: Dataframes for edit distance, canonicity, mask analysis, and perplexity plots
    """
    # Edit Distance dataframe
    df_edit = df.dropna(subset=["Step Number", "Edit Distance"]).copy()
    df_edit.sort_values("Step Number", inplace=True)
    
    # Canonicity dataframe
    df_canonicity = df.dropna(subset=["Step Number", "Sample Index", "Canonical?"]).copy()
    df_canonicity.sort_values(["Sample Index", "Step Number"], inplace=True)
    
    # Mask analysis dataframe
    df_mask = None
    if "Decoded Text" in df.columns:
        df_mask = df.copy()
        df_mask["Mask Count"] = df_mask["Decoded Text"].apply(
            lambda text: text.count("[MASK]") if isinstance(text, str) else 0
        )
        df_mask.dropna(subset=["Step Number", "Sample Index"], inplace=True)
        df_mask.sort_values(["Sample Index", "Step Number"], inplace=True)
    
    # Perplexity dataframe
    df_perplex = df.dropna(subset=["Step Number", "Original Perplexity", "Retokenized Perplexity"]).copy()
    df_perplex.sort_values("Step Number", inplace=True)
    
    return df_edit, df_canonicity, df_mask, df_perplex

# -------------------------------
# Plotting Functions
# -------------------------------
def plot_edit_distance(df_edit):
    """
    Generates a plot showing the average edit distance across diffusion steps.
    
    Args:
        df_edit (pd.DataFrame): Dataframe containing edit distance data
    """
    grouped = df_edit.groupby("Step Number")["Edit Distance"].mean().reset_index()
    grouped.sort_values("Step Number", inplace=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(grouped["Step Number"], grouped["Edit Distance"], 
             color='blue', linewidth=2, label="Avg Edit Distance")
    plt.xlabel("Step Number")
    plt.ylabel("Average Edit Distance")
    plt.title("Average Edit Distance vs. Step Number")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('inter_avg_edit_distance.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_transition_and_canonicity(df_canonicity):
    """
    Generates a dual-axis plot showing transition frequency and percent canonicity.
    
    Args:
        df_canonicity (pd.DataFrame): Dataframe containing canonicity data
    """
    # Calculate transition steps
    transition_steps = []
    for sample_index, group in df_canonicity.groupby("Sample Index"):
        group = group.copy()
        group["Prev_Canonical"] = group["Canonical?"].shift(1)
        transition_rows = group[(group["Canonical?"] == 0) & (group["Prev_Canonical"] == 1)]
        if not transition_rows.empty:
            transition_steps.append(transition_rows.iloc[0]["Step Number"])
    
    # Calculate transition frequencies
    transition_counter = Counter(transition_steps)
    transition_steps_sorted = sorted(transition_counter.keys())
    freqs = [transition_counter[s] for s in transition_steps_sorted]
    
    # Calculate percent canonicity
    df_steps = df_canonicity[df_canonicity["Step Number"] <= 1025].copy()
    canonicity_by_step = df_steps.groupby("Step Number")["Canonical?"].mean() * 100
    canonicity_by_step = canonicity_by_step.reset_index()
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot transition frequency (bar chart)
    ax1.bar(transition_steps_sorted, freqs, width=1.0, color='blue', alpha=0.7, 
            label="Transition Frequency")
    ax1.set_xlabel("Step Number")
    ax1.set_ylabel("Frequency of Transition", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)
    
    # Plot percent canonicity (line chart)
    ax2 = ax1.twinx()
    ax2.plot(canonicity_by_step["Step Number"], canonicity_by_step["Canonical?"], 
             color='red', linewidth=2, label="Percent Canonicity")
    ax2.set_ylabel("Percent Canonicity (%)", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    
    # Set plot limits and legend
    ax1.set_xlim(0, 1025)
    ax2.set_xlim(0, 1025)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title("Transition Frequency and Percent Canonicity vs. Step Number")
    plt.tight_layout()
    plt.savefig("inter_transition_and_canonicity_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_mask_transition_and_count(df_mask):
    """
    Generates a dual-axis plot showing [MASK] token transitions and counts.
    
    Args:
        df_mask (pd.DataFrame): Dataframe containing mask token data
    """
    if df_mask is None:
        return
        
    # Calculate average [MASK] count per step
    avg_mask_by_step = df_mask.groupby("Step Number")["Mask Count"].mean().reset_index()
    avg_mask_by_step.sort_values("Step Number", inplace=True)
    
    # Calculate transition steps for [MASK] tokens
    transition_steps = []
    for sample_index, group in df_mask.groupby("Sample Index"):
        group = group.copy().sort_values("Step Number")
        group["Prev_Mask_Count"] = group["Mask Count"].shift(1)
        transition = group[(group["Mask Count"] == 0) & (group["Prev_Mask_Count"] > 0)]
        if not transition.empty:
            transition_steps.append(transition.iloc[0]["Step Number"])
    
    # Calculate transition frequencies
    transition_counter = Counter(transition_steps)
    transition_steps_sorted = sorted(transition_counter.keys())
    transition_freqs = [transition_counter[s] for s in transition_steps_sorted]
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot transition frequency (bar chart)
    ax1.bar(transition_steps_sorted, transition_freqs, width=1.0, color='blue', alpha=0.7, 
            label="Transition Frequency")
    ax1.set_xlabel("Step Number")
    ax1.set_ylabel("Frequency of Transition (Last [MASK] eliminated)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)
    
    # Plot average [MASK] count (line chart)
    ax2 = ax1.twinx()
    ax2.plot(avg_mask_by_step["Step Number"], avg_mask_by_step["Mask Count"], 
             color='red', linewidth=2, label="Avg [MASK] Count")
    ax2.set_ylabel("Average [MASK] Count", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    
    # Set plot limits and legend
    if transition_steps_sorted:
        max_step = max(transition_steps_sorted)
        ax1.set_xlim(0, max_step)
        ax2.set_xlim(0, max_step)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title("Average [MASK] Count and Transition Frequency vs. Step Number")
    plt.tight_layout()
    plt.savefig("inter_mask_transition_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_perplexities(df_perplex):
    """
    Generates a plot comparing original and retokenized perplexities.
    
    Args:
        df_perplex (pd.DataFrame): Dataframe containing perplexity data
    """
    grouped = df_perplex.groupby("Step Number")[["Original Perplexity", "Retokenized Perplexity"]].mean().reset_index()
    grouped.sort_values("Step Number", inplace=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(grouped["Step Number"], grouped["Original Perplexity"], 
             color="blue", linewidth=2, label="Original Perplexity")
    plt.plot(grouped["Step Number"], grouped["Retokenized Perplexity"], 
             color="red", linewidth=2, label="Retokenized Perplexity")
    plt.xlabel("Step Number")
    plt.ylabel("Perplexity")
    plt.title("Average Original and Retokenized Perplexities vs. Step Number")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inter_perplexities_vs_steps.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_perplexity_difference(df_perplex):
    """
    Generates a plot showing the difference between original and retokenized perplexities.
    
    Args:
        df_perplex (pd.DataFrame): Dataframe containing perplexity data
    """
    grouped = df_perplex.groupby("Step Number")[["Original Perplexity", "Retokenized Perplexity"]].mean().reset_index()
    grouped.sort_values("Step Number", inplace=True)
    
    # Calculate perplexity difference
    diff = grouped["Original Perplexity"] - grouped["Retokenized Perplexity"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(grouped["Step Number"], diff, color="green", linewidth=2, 
             label="Perplexity Difference")
    plt.xlabel("Step Number")
    plt.ylabel("Perplexity Difference (Original - Retokenized)")
    plt.title("Average Perplexity Difference vs. Step Number")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inter_perplexity_difference_vs_steps.png", bbox_inches='tight', dpi=300)
    plt.close()

# -------------------------------
# Main Execution
# -------------------------------
def main():
    """
    Main function to load data and generate all plots.
    """
    # Load and preprocess data
    df = load_and_preprocess_data()
    df_edit, df_canonicity, df_mask, df_perplex = prepare_plot_dataframes(df)
    
    # Generate plots
    plot_edit_distance(df_edit)
    plot_transition_and_canonicity(df_canonicity)
    plot_mask_transition_and_count(df_mask)
    plot_perplexities(df_perplex)
    plot_perplexity_difference(df_perplex)

if __name__ == "__main__":
    main()