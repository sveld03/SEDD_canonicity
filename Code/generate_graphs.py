import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

plt.switch_backend('Agg')

# Load the CSV file
file_path = "Big_Data/intermediate-data.csv"
df = pd.read_csv(file_path, low_memory=False)

# Ensure relevant columns are numeric
df["Step Number"] = pd.to_numeric(df["Step Number"], errors="coerce")
df["Sample Index"] = pd.to_numeric(df["Sample Index"], errors="coerce")
df.dropna(subset=["Step Number", "Sample Index"], inplace=True)

# Create a new column "Mask Count" by counting "[MASK]" occurrences in the "Decoded Text" column
if "Decoded Text" in df.columns:
    df["Mask Count"] = df["Decoded Text"].apply(lambda text: text.count("[MASK]") if isinstance(text, str) else 0)
else:
    print("Column 'Decoded Text' not found.")
    exit()

# -------------------------------------------------------------------
# Part A: Average number of "[MASK]" tokens at each step
# -------------------------------------------------------------------
avg_mask_by_step = df.groupby("Step Number")["Mask Count"].mean().reset_index()
avg_mask_by_step = avg_mask_by_step.sort_values("Step Number")

# -------------------------------------------------------------------
# Part B: Frequency of transition steps
# A transition is defined as the step where the last "[MASK]" token is eliminated.
# For each sample, find the first step where the mask count becomes 0, having been >0 on the previous step.
# -------------------------------------------------------------------
transition_steps = []
for sample_index, group in df.groupby("Sample Index"):
    group = group.copy().sort_values("Step Number")
    group["Prev_Mask_Count"] = group["Mask Count"].shift(1)
    transition = group[(group["Mask Count"] == 0) & (group["Prev_Mask_Count"] > 0)]
    if not transition.empty:
        transition_step = transition.iloc[0]["Step Number"]
        transition_steps.append(transition_step)

# Count frequency of these transition steps
transition_counter = Counter(transition_steps)
transition_steps_sorted = sorted(transition_counter.keys())
transition_freqs = [transition_counter[s] for s in transition_steps_sorted]

# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for transition frequency on the primary y-axis (blue)
ax1.bar(transition_steps_sorted, transition_freqs, width=1.0, color='blue', alpha=0.7, label="Transition Frequency")
ax1.set_xlabel("Step Number")
ax1.set_ylabel("Frequency of Transition (Last [MASK] eliminated)", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.grid(True)

# Secondary y-axis: Average [MASK] count (red line)
ax2 = ax1.twinx()
ax2.plot(avg_mask_by_step["Step Number"], avg_mask_by_step["Mask Count"], color='red', linewidth=2, label="Avg [MASK] Count")
ax2.set_ylabel("Average [MASK] Count", color="red")
ax2.tick_params(axis='y', labelcolor="red")

# Limit x-axis to the range where transitions occur
if transition_steps_sorted:
    max_step = max(transition_steps_sorted)
    ax1.set_xlim(0, max_step)
    ax2.set_xlim(0, max_step)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title("Average [MASK] Count and Transition Frequency vs. Step Number")
plt.tight_layout()
plt.savefig("mask_transition_frequency.png", dpi=300, bbox_inches='tight')