import matplotlib.pyplot as plt

# Ensure the correct backend
plt.switch_backend('Agg')

# Data from the file
tokens_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
steps = [200, 242, 284, 326, 368, 410, 452, 494, 536, 578, 620, 662, 704, 746, 788, 830, 872, 914, 956, 998]
percent_canonicity = [78.0, 78.0, 70.0, 68.0, 56.0, 38.0, 36.0, 40.0, 30.0, 42.0, 44.0, 32.0, 22.0, 36.0, 18.0, 28.0, 34.0, 24.0, 24.0, 26.0]
avg_edit_distance = [0.72, 1.56, 1.18, 1.76, 2.84, 2.4, 2.48, 3.7, 4.06, 3.48, 2.4, 4.72, 4.06, 4.0, 5.2, 2.88, 3.84, 4.34, 6.22, 6.64]
non_canonical_log_likelihood = [-197.63104736328125, -383.9724090576172, -569.4825503540039, -782.4281646728516, -1019.4935388183594, -1158.107095336914, -1324.0048291015626, -1571.6267602539062, -1751.367958984375, -1931.927412109375, -2121.619912109375, -2374.999377441406, -2506.68126953125, -2691.5863647460938, -2863.00134765625, -3141.4494921875, -3213.7533349609375, -3357.913203125, -3691.9137548828126, -3904.264833984375]
canonical_log_likelihood = [-199.85525939941405, -392.708359375, -573.5283023071289, -787.3895135498046, -1025.7165087890626, -1168.4060382080079, -1330.4199487304688, -1583.5249072265624, -1765.6376635742188, -1939.5368920898438, -2129.3593115234376, -2390.315842285156, -2520.362451171875, -2703.46859375, -2884.36662109375, -3153.4573876953127, -3228.093369140625, -3375.6464794921876, -3714.0429248046876, -3924.0240283203125]

# Calculate the difference between non-canonical and canonical log-likelihood
log_likelihood_difference = [non - can for non, can in zip(non_canonical_log_likelihood, canonical_log_likelihood)]

# Create the plot with larger figure size
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot percent canonicity on the first y-axis
color_canonicity = 'tab:blue'
ax1.set_xlabel('Number of Tokens')
ax1.set_ylabel('Percent Canonicity', color=color_canonicity)
line1 = ax1.plot(tokens_steps, percent_canonicity, marker='o', color=color_canonicity, label='Percent Canonicity')
ax1.tick_params(axis='y', labelcolor=color_canonicity)

# Create second y-axis for average edit distance
ax2 = ax1.twinx()
color_edit_distance = 'tab:green'
ax2.set_ylabel('Average Edit Distance', color=color_edit_distance)
line2 = ax2.plot(tokens_steps, avg_edit_distance, marker='s', color=color_edit_distance, label='Average Edit Distance')
ax2.tick_params(axis='y', labelcolor=color_edit_distance)

# Create third y-axis for log-likelihood difference
ax3 = ax1.twinx()
# Move the third axis spine outward
ax3.spines['right'].set_position(('outward', 60))
color_difference = 'tab:red'
ax3.set_ylabel('Log-Likelihood Difference', color=color_difference)
line3 = ax3.plot(tokens_steps, log_likelihood_difference, marker='^', color=color_difference, label='Log-Likelihood Difference')
ax3.tick_params(axis='y', labelcolor=color_difference)

# Add second x-axis for step count
ax4 = ax1.twiny()
ax4.set_xlim(ax1.get_xlim())
ax4.set_xticks(tokens_steps[::2])
ax4.set_xticklabels(steps[::2])
ax4.set_xlabel('Step Count')

# Set x-ticks for token count
ax1.set_xticks(tokens_steps[::2])
ax1.set_xticklabels(tokens_steps[::2])

# Add legend
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot to a file
plt.savefig('combined_analysis_plot_2.png', bbox_inches='tight', dpi=300)

# Show the plot (optional, might not work in headless mode)
plt.show()