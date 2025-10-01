
# %%
import numpy as np
import matplotlib.pyplot as plt

# Generate data - centered around 0 for better symmetry
x = np.linspace(-10, 10, 200)

# Create figure with better aspect ratio
fig, ax = plt.subplots(figsize=(10, 6))

# Warm/cool color palette - warm for positive β, cool for negative β  
colors = ['#d73027', '#fc8d59', '#fee08b', '#e0f3f8', '#91bfdb', '#4575b4']
beta_values = [2, 1, 0.5, -0.5, -1, -2]
line_widths = [3, 2.5, 2, 2, 2.5, 3]  # Vary thickness by magnitude

# Plot curves with varied line widths
for i, beta in enumerate(beta_values):
    p = np.exp(beta * x) / (1 + np.exp(beta * x))
    ax.plot(x, p, linewidth=line_widths[i], color=colors[i], 
            label=f'β = {beta}', alpha=0.9)

# Remove top and right spines (Tufte's principle)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Lighten remaining spines
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# Customize grid - subtle and minimal
ax.grid(True, linestyle='-', alpha=0.2, color='gray')
ax.set_axisbelow(True)

# Set axis labels with better typography
ax.set_xlabel('x', fontsize=14, color='#333333')
ax.set_ylabel('P(Y = 1 | X = x)', fontsize=14, color='#333333')

# Customize tick parameters
ax.tick_params(axis='both', which='major', labelsize=12, colors='#666666')
ax.tick_params(axis='both', which='major', length=0)  # Remove tick marks

# Enhanced legend with better grouping
handles, labels = ax.get_legend_handles_labels()
# Reorder to group positive and negative values
pos_indices = [0, 1, 2]  # β = 2, 1, 0.5
neg_indices = [3, 4, 5]  # β = -0.5, -1, -2

# Create two-column legend
legend1 = ax.legend([handles[i] for i in pos_indices], 
                   [labels[i] for i in pos_indices],
                   loc='upper left', bbox_to_anchor=(1, 1), 
                   fontsize=11, frameon=False, title='Positive β')
ax.add_artist(legend1)
legend2 = ax.legend([handles[i] for i in neg_indices], 
                   [labels[i] for i in neg_indices],
                   loc='lower left', bbox_to_anchor=(1, 0), 
                   fontsize=11, frameon=False, title='Negative β')

# Add key reference lines
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1)
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.6, linewidth=1)

# Set symmetric axis limits for better visual balance
ax.set_xlim(-10, 10)
ax.set_ylim(-0.02, 1.02)

# Add subtle annotations for key points
ax.text(0.2, 0.52, 'Decision\nThreshold', fontsize=10, color='gray', 
        ha='left', va='bottom', alpha=0.8)
ax.text(0.2, 0.02, 'Inflection Point', fontsize=10, color='gray',
        ha='left', va='bottom', alpha=0.8)

# Add subtle title if needed (uncomment if desired)
# ax.set_title('Logistic Function for Different β Values', 
#              fontsize=16, color='#333333', pad=20)

# Optimize layout
plt.tight_layout()

# Save with high DPI for publication quality
# plt.savefig('logistic_curves_tufte.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('logistic_curves_tufte.png', dpi=300, bbox_inches='tight')

plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 200)
fig, ax = plt.subplots(figsize=(3.5, 2.2))

colors = ['#d73027', '#fc8d59', '#fee08b', '#e0f3f8', '#91bfdb', '#4575b4']
beta_values = [2, 1, 0.5, -0.5, -1, -2]
line_widths = [1.5, 1.2, 1, 1, 1.2, 1.5]

beta0 = 1
for i, beta in enumerate(beta_values):
    linear = beta0 + beta * x
    p = np.exp(linear) / (1 + np.exp(linear))
    ax.plot(x, p, linewidth=line_widths[i], color=colors[i], 
            label=f'β = {beta}', alpha=0.9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.grid(True, linestyle='--', alpha=0.2, color='k', linewidth=0.25)
ax.set_axisbelow(True)
ax.set_xlabel('x', fontsize=10, color='#333333')
ax.set_ylabel('P(Y = 1 | X = x)', fontsize=10, color='#333333')
ax.tick_params(axis='both', which='major', labelsize=8, colors='#666666')
ax.tick_params(axis='both', which='major', length=0)  # Remove tick marks
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5), 
          fontsize=8, frameon=False, labelspacing=0.3,
          handlelength=0.75, handletextpad=0.5)

ax.set_xlim(-10, 10)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()

# plt.savefig('logistic_curves_journal.pdf', dpi=300, bbox_inches='tight', 
#             facecolor='white', edgecolor='none')

# %%
