import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("benchmark_results.csv")

# Calculate Speedup Multiplier
df['Speedup_Factor'] = df['TPS'] / df.groupby('Prompt_ID')['TPS'].transform('min')
# Filter to just Medusa so we only plot the final speedup
df_medusa = df[df['Mode'] == 'Medusa'].copy()

# Set the style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 7))

# Create a bar plot for the TPS, colored by whether it was an Exact Match
ax = sns.barplot(
    data=df_medusa, 
    x='Category', 
    y='TPS', 
    hue='Exact_Match',
    dodge=False, # Keep bars centered
    palette={True: '#2ecc71', False: '#e74c3c'} # Green for match, Red for drift
)

# Add a horizontal line representing the average Baseline TPS (~81 TPS)
plt.axhline(y=81.0, color='#34495e', linestyle='--', label='Average Baseline TPS (~81)')

# Annotate each bar with its Speedup Multiplier
for i, row in df_medusa.reset_index().iterrows():
    ax.text(
        i, 
        row['TPS'] + 1, 
        f"{row['Speedup_Factor']:.2f}x", 
        color='black', 
        ha="center", 
        fontweight='bold'
    )

# Formatting
plt.title('Medusa Speculative Decoding: TPS, Speedup, and FP16 Divergence', fontsize=16, pad=20)
plt.ylabel('Tokens Per Second (TPS)', fontsize=12)
plt.xlabel('Prompt Category', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Custom Legend
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=['Diverged (False)', 'Exact Match (True)', 'Baseline Baseline'], title='Output Integrity', loc='upper right')

plt.tight_layout()
plt.savefig('advanced_medusa_benchmark.png', dpi=300)
print("Advanced chart saved as advanced_medusa_benchmark.png!")