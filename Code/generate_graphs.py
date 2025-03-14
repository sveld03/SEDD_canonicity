import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

plt.switch_backend('Agg')

# -------------------------------
# 1. Load and Preprocess the Data
# -------------------------------
file_path = "intermediate-data.csv"
df = pd.read_csv(file_path, low_memory=False)

# Define the columns that should be numeric.
numeric_columns = ["Step Number", "Edit Distance", "Sample Index", 
                   "Canonical?", "Original Perplexity", "Retokenized Perplexity"]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# 2. Create DataFrames for Each Plot
# -------------------------------
# Data for Edit Distance plot
df_edit = df.dropna(subset=["Step Number", "Edit Distance"]).copy()
df_edit.sort_values("Step Number", inplace=True)

# Data for Transition Frequency & Percent Canonicity plot
df_canonicity = df.dropna(subset=["Step Number", "Sample Index", "Canonical?"]).copy()
df_canonicity.sort_values(["Sample Index", "Step Number"], inplace=True)

# Data for [MASK] Count plots (requires "Decoded Text")
if "Decoded Text" in df.columns:
    df_mask = df.copy()
    df_mask["Mask Count"] = df_mask["Decoded Text"].apply(lambda text: text.count("[MASK]") 
                                                          if isinstance(text, str) else 0)
    df_mask.dropna(subset=["Step Number", "Sample Index"], inplace=True)
    df_mask.sort_values(["Sample Index", "Step Number"], inplace=True)
else:
    df_mask = None

# Data for Perplexity plots
df_perplex = df.dropna(subset=["Step Number", "Original Perplexity", "Retokenized Perplexity"]).copy()
df_perplex.sort_values("Step Number", inplace=True)

# -------------------------------
# 3. Define Plotting Functions
# -------------------------------

def plot_edit_distance(df_edit):
    grouped = df_edit.groupby("Step Number")["Edit Distance"].mean().reset_index()
    grouped.sort_values("Step Number", inplace=True)
    steps = grouped["Step Number"]
    avg_edit_distance = grouped["Edit Distance"]
    
    plt.figure(figsize=(8, 6))
    plt.plot(steps, avg_edit_distance, color='blue', linewidth=2, label="Avg Edit Distance")
    plt.xlabel("Step Number")
    plt.ylabel("Average Edit Distance")
    plt.title("Average Edit Distance vs. Step Number")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('inter_avg_edit_distance.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_transition_and_canonicity(df_canonicity):
    # Part 1: Frequency of Transition Steps
    transition_steps = []
    for sample_index, group in df_canonicity.groupby("Sample Index"):
        group = group.copy()
        group["Prev_Canonical"] = group["Canonical?"].shift(1)
        transition_rows = group[(group["Canonical?"] == 0) & (group["Prev_Canonical"] == 1)]
        if not transition_rows.empty:
            transition_steps.append(transition_rows.iloc[0]["Step Number"])
    
    transition_counter = Counter(transition_steps)
    transition_steps_sorted = sorted(transition_counter.keys())
    freqs = [transition_counter[s] for s in transition_steps_sorted]
    
    # Part 2: Percent Canonicity (up to step 1025)
    df_steps = df_canonicity[df_canonicity["Step Number"] <= 1025].copy()
    canonicity_by_step = df_steps.groupby("Step Number")["Canonical?"].mean() * 100
    canonicity_by_step = canonicity_by_step.reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(transition_steps_sorted, freqs, width=1.0, color='blue', alpha=0.7, label="Transition Frequency")
    ax1.set_xlabel("Step Number")
    ax1.set_ylabel("Frequency of Transition", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.set_title("Transition Frequency and Percent Canonicity vs. Step Number")
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(canonicity_by_step["Step Number"], canonicity_by_step["Canonical?"], 
             color='red', linewidth=2, label="Percent Canonicity")
    ax2.set_ylabel("Percent Canonicity (%)", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    
    ax1.set_xlim(0, 1025)
    ax2.set_xlim(0, 1025)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig("inter_transition_and_canonicity_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_mask_transition_and_count(df_mask):
    if df_mask is None:
        return
    # Part A: Average [MASK] Count
    avg_mask_by_step = df_mask.groupby("Step Number")["Mask Count"].mean().reset_index()
    avg_mask_by_step.sort_values("Step Number", inplace=True)
    
    # Part B: Transition Frequency for [MASK] tokens
    transition_steps = []
    for sample_index, group in df_mask.groupby("Sample Index"):
        group = group.copy().sort_values("Step Number")
        group["Prev_Mask_Count"] = group["Mask Count"].shift(1)
        transition = group[(group["Mask Count"] == 0) & (group["Prev_Mask_Count"] > 0)]
        if not transition.empty:
            transition_steps.append(transition.iloc[0]["Step Number"])
    
    transition_counter = Counter(transition_steps)
    transition_steps_sorted = sorted(transition_counter.keys())
    transition_freqs = [transition_counter[s] for s in transition_steps_sorted]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(transition_steps_sorted, transition_freqs, width=1.0, color='blue', alpha=0.7, 
            label="Transition Frequency")
    ax1.set_xlabel("Step Number")
    ax1.set_ylabel("Frequency of Transition (Last [MASK] eliminated)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(avg_mask_by_step["Step Number"], avg_mask_by_step["Mask Count"], 
             color='red', linewidth=2, label="Avg [MASK] Count")
    ax2.set_ylabel("Average [MASK] Count", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    
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
    grouped = df_perplex.groupby("Step Number")[["Original Perplexity", "Retokenized Perplexity"]].mean().reset_index()
    grouped.sort_values("Step Number", inplace=True)
    steps = grouped["Step Number"]
    avg_orig = grouped["Original Perplexity"]
    avg_retoken = grouped["Retokenized Perplexity"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, avg_orig, color="blue", linewidth=2, label="Original Perplexity")
    plt.plot(steps, avg_retoken, color="red", linewidth=2, label="Retokenized Perplexity")
    plt.xlabel("Step Number")
    plt.ylabel("Perplexity")
    plt.title("Average Original and Retokenized Perplexities vs. Step Number")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inter_perplexities_vs_steps.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_perplexity_difference(df_perplex):
    grouped = df_perplex.groupby("Step Number")[["Original Perplexity", "Retokenized Perplexity"]].mean().reset_index()
    grouped.sort_values("Step Number", inplace=True)
    steps = grouped["Step Number"]
    avg_orig = grouped["Original Perplexity"]
    avg_retoken = grouped["Retokenized Perplexity"]
    diff = avg_orig - avg_retoken
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, diff, color="green", linewidth=2, label="Perplexity Difference")
    plt.xlabel("Step Number")
    plt.ylabel("Perplexity Difference (Original - Retokenized)")
    plt.title("Average Perplexity Difference vs. Step Number")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inter_perplexity_difference_vs_steps.png", bbox_inches='tight', dpi=300)
    plt.close()

# -------------------------------
# 4. Generate All Graphs
# -------------------------------
# plot_edit_distance(df_edit)
plot_transition_and_canonicity(df_canonicity)
# plot_mask_transition_and_count(df_mask)
# plot_perplexities(df_perplex)
# plot_perplexity_difference(df_perplex)