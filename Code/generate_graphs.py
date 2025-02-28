import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

plt.switch_backend('Agg')

# Load the CSV file
file_path = "Big_Data/intermediate-data.csv"
df = pd.read_csv(file_path, low_memory=False)

# Convert relevant columns to numeric
df["Step Number"] = pd.to_numeric(df["Step Number"], errors="coerce")
df["Sample Index"] = pd.to_numeric(df["Sample Index"], errors="coerce")
df["Canonical?"] = pd.to_numeric(df["Canonical?"], errors="coerce")

# Drop rows with missing values in these columns
df.dropna(subset=["Step Number", "Sample Index", "Canonical?"], inplace=True)

# Sort by sample index and step number
df.sort_values(["Sample Index", "Step Number"], inplace=True)

### Part 1: Frequency of Transition Steps

# Identify the first step at which each sample transitions from canonical (1) to non-canonical (0)
transition_steps = []
for sample_index, group in df.groupby("Sample Index"):
    group = group.copy()
    group["Prev_Canonical"] = group["Canonical?"].shift(1)
    transition_rows = group[(group["Canonical?"] == 0) & (group["Prev_Canonical"] == 1)]
    if transition_rows.empty:
        continue  # Skip samples that never transition
    transition_step = transition_rows.iloc[0]["Step Number"]
    transition_steps.append(transition_step)

# Count frequency of transition steps
transition_counter = Counter(transition_steps)
transition_steps_sorted = sorted(transition_counter.keys())
freqs = [transition_counter[s] for s in transition_steps_sorted]

### Part 2: Percent Canonicity

# For each step (up to 350) compute the percent of samples that are fully canonical
# (i.e., "Canonical?" == 1)
df_steps = df[df["Step Number"] <= 350].copy()
canonicity_by_step = df_steps.groupby("Step Number")["Canonical?"].mean() * 100
canonicity_by_step = canonicity_by_step.reset_index()

### Plotting

fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for transition frequency
ax1.bar(transition_steps_sorted, freqs, width=1.0, color='blue', alpha=0.7, label="Transition Frequency")
ax1.set_xlabel("Step Number")
ax1.set_ylabel("Frequency of Transition", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.set_title("Transition Frequency and Percent Canonicity vs. Step Number")
ax1.grid(True)

# Create a second y-axis for percent canonicity
ax2 = ax1.twinx()
ax2.plot(canonicity_by_step["Step Number"], canonicity_by_step["Canonical?"], color='red', linewidth=2, label="Percent Canonicity")
ax2.set_ylabel("Percent Canonicity (%)", color="red")
ax2.tick_params(axis='y', labelcolor="red")

# Limit the x-axis to 0 to 350 since there are no transitions beyond that point
ax1.set_xlim(0, 350)
ax2.set_xlim(0, 350)

# Optional: add legends. Since we have two axes, we need to combine them:
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig("transition_and_canonicity_frequency.png", dpi=300, bbox_inches='tight')