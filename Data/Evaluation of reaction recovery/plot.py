import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
file_path = "HGLMA_recovery.xlsx - Sheet1.csv"
df = pd.read_csv(file_path)

# 2. Data Preprocessing
# The file has repeated columns (Top25, Top25.1, etc.), we need to merge them
# Extract first set of columns
part1 = df[['Top25', 'Top50', 'Top100', 'TopN']].copy()
# Extract second set of columns and rename them to match
part2 = df[['Top25.1', 'Top50.1', 'Top100.1', 'TopN.1']].copy()
part2.columns = ['Top25', 'Top50', 'Top100', 'TopN']

# Concatenate vertically
combined_df = pd.concat([part1, part2], ignore_index=True)

# Transform to long format for plotting
plot_data = combined_df.melt(var_name='Metric', value_name='Value')

# 3. Plotting
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")

# Boxplot without outliers (showfliers=False)
sns.boxplot(x='Metric', y='Value', data=plot_data,
            order=['Top25', 'Top50', 'Top100', 'TopN'],
            width=0.5, showfliers=False)

# Settings
plt.ylim(0.4, 1.0)  # Range 0.4 - 1.0
plt.ylabel('Performance')
plt.xlabel('')
plt.title('HGLMA_recovery')

# Save
plt.savefig('recovery_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()