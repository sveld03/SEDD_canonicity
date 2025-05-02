# This script was used to generate Old_Graphs/log_likelihoods_classified.png

"""
Script for generating a plot showing log-likelihoods for canonical and non-canonical sequences
across different token counts. The plot includes a secondary x-axis showing step counts.
"""

import matplotlib.pyplot as plt

# Set matplotlib backend to non-interactive mode for server environments
plt.switch_backend('Agg')

# -------------------------------
# Data Configuration
# -------------------------------
# Hard-coded data for token counts, step counts, and log-likelihoods
data = {
    'Token Count': [100, 250, 400, 500, 750, 900, 1000],
    'Step Count': [250, 400, 500, 600, 750, 900, 1000],
    'Canonical': [-382.3238701, -978.72572, -1518.879805, -1924.41208, -2761.688437, -3366.333885, -3770.884953],
    'Non-canonical': [-398.4841293, -985.8946772, -1600.329771, -1997.823681, -2925.044833, -3521.048261, -3820.884584]
}

# -------------------------------
# Plot Generation
# -------------------------------
# Create figure with primary y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot log-likelihoods for both canonical and non-canonical sequences
ax1.plot(data['Token Count'], data['Canonical'], label='Canonical', marker='o')
ax1.plot(data['Token Count'], data['Non-canonical'], label='Non-canonical', marker='o')

# Configure primary axis labels and title
ax1.set_xlabel('Token Count')
ax1.set_ylabel('Log-Likelihood')
ax1.set_title('Log-Likelihoods for Canonical and Non-canonical Sequences')
ax1.legend()

# Set x-ticks to match data points
ax1.set_xticks(data['Token Count'])

# -------------------------------
# Secondary Axis Setup
# -------------------------------
# Create secondary x-axis for Step Count
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(data['Token Count'])
ax2.set_xticklabels(data['Step Count'])
ax2.set_xlabel('Step Count')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('log_likelihoods_classified.png', bbox_inches='tight', dpi=300)

# Commented out code for additional analysis
# # Data from the file
# tokens_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
# steps = [200, 242, 284, 326, 368, 410, 452, 494, 536, 578, 620, 662, 704, 746, 788, 830, 872, 914, 956, 998]
# percent_canonicity = [78.0, 78.0, 70.0, 68.0, 56.0, 38.0, 36.0, 40.0, 30.0, 42.0, 44.0, 32.0, 22.0, 36.0, 18.0, 28.0, 34.0, 24.0, 24.0, 26.0]
# avg_edit_distance = [0.72, 1.56, 1.18, 1.76, 2.84, 2.4, 2.48, 3.7, 4.06, 3.48, 2.4, 4.72, 4.06, 4.0, 5.2, 2.88, 3.84, 4.34, 6.22, 6.64]
# non_canonical_log_likelihood = [-197.63104736328125, -383.9724090576172, -569.4825503540039, -782.4281646728516, -1019.4935388183594, -1158.107095336914, -1324.0048291015626, -1571.6267602539062, -1751.367958984375, -1931.927412109375, -2121.619912109375, -2374.999377441406, -2506.68126953125, -2691.5863647460938, -2863.00134765625, -3141.4494921875, -3213.7533349609375, -3357.913203125, -3691.9137548828126, -3904.264833984375]
# canonical_log_likelihood = [-199.85525939941405, -392.708359375, -573.5283023071289, -787.3895135498046, -1025.7165087890626, -1168.4060382080079, -1330.4199487304688, -1583.5249072265624, -1765.6376635742188, -1939.5368920898438, -2129.3593115234376, -2390.315842285156, -2520.362451171875, -2703.46859375, -2884.36662109375, -3153.4573876953127, -3228.093369140625, -3375.6464794921876, -3714.0429248046876, -3924.0240283203125]