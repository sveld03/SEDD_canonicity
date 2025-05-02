import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# Example data: Replace with your actual log-likelihoods
data = {
    100: {"canonical": [1.2, 1.5, 1.3, 1.4], "non_canonical": [1.0, 1.1, 0.9, 1.2]},
    200: {"canonical": [1.8, 1.7, 1.6, 1.9], "non_canonical": [1.4, 1.5, 1.3, 1.6]},
    # Add more sequence lengths and data...
}

alpha = 0.05  # Significance level
results = []

for length, groups in data.items():
    canonical = np.array(groups["canonical"])
    non_canonical = np.array(groups["non_canonical"])
    
    # Perform t-test
    t_stat, p_value_t = ttest_ind(canonical, non_canonical, equal_var=False)
    
    # Perform Mann-Whitney U test
    u_stat, p_value_u = mannwhitneyu(canonical, non_canonical, alternative="two-sided")
    
    # Store results
    results.append({
        "Sequence Length": length,
        "Mean Canonical": np.mean(canonical),
        "Mean Non-Canonical": np.mean(non_canonical),
        "t-test p-value": p_value_t,
        "Mann-Whitney U p-value": p_value_u
    })

# Print results
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)
