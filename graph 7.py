import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter1d

# Load the data
file_path = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"
performance_df = pd.read_excel(file_path, sheet_name='Performance')

# Filter for Amazon UK and ASOS
companies = ['Amazon UK', 'ASOS']
perf_data = performance_df[performance_df['company'].isin(companies)].copy()

# Select key metrics for parallel coordinates
metrics = [
    'on_time_delivery_rate_pct',
    'customer_satisfaction_avg',
    'nps_score',
    'first_attempt_success_rate_pct',
    'positive_sentiment_pct',
    'negative_sentiment_pct',
    'cost_per_delivery_gbp'
]

metric_labels = [
    'On-Time\nDelivery %',
    'Customer\nSatisfaction',
    'NPS\nScore',
    'First-Attempt\nSuccess %',
    'Positive\nSentiment %',
    'Negative\nSentiment %',
    'Cost per\nDelivery (£)'
]

# Aggregate data by company
company_data = perf_data.groupby('company')[metrics].mean().reset_index()

# Normalize data to 0-1 scale for parallel coordinates
normalized_data = company_data.copy()
for metric in metrics:
    min_val = company_data[metric].min()
    max_val = company_data[metric].max()
    if max_val - min_val > 0:
        normalized_data[metric] = (company_data[metric] - min_val) / (max_val - min_val)
    else:
        normalized_data[metric] = 0.5

# Create figure
fig, ax = plt.subplots(figsize=(18, 10))
fig.patch.set_facecolor('white')

# Set up x-axis positions
x_positions = np.arange(len(metrics))

# Define colors with transparency
colors = {
    'Amazon UK': ('#0066CC', 0.7),
    'ASOS': ('#FF6B35', 0.7)
}

# Plot parallel coordinates with blur gradient effect
for idx, row in normalized_data.iterrows():
    company = row['company']
    values = [row[metric] for metric in metrics]
    
    # Apply gaussian smoothing for blur effect
    smooth_values = gaussian_filter1d(values, sigma=0.3)
    
    color, alpha = colors[company]
    
    # Plot multiple offset lines for blur gradient effect
    offsets = np.linspace(-0.02, 0.02, 5)
    alphas = np.linspace(alpha * 0.3, alpha, 5)
    
    for offset, alpha_val in zip(offsets, alphas):
        offset_values = smooth_values + offset
        ax.plot(x_positions, offset_values, 
               color=color, alpha=alpha_val, linewidth=3.5,
               marker='o', markersize=8, markeredgewidth=0)
    
    # Main line
    ax.plot(x_positions, smooth_values, 
           color=color, alpha=alpha, linewidth=4,
           marker='o', markersize=10, markeredgewidth=2,
           markeredgecolor='white', label=company, zorder=10)

# Customize axes
ax.set_xlim(-0.5, len(metrics) - 0.5)
ax.set_ylim(-0.1, 1.1)
ax.set_xticks(x_positions)
ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['Low', '', 'Medium', '', 'High'], fontsize=10)

# Add vertical grid lines with gradient
for x in x_positions:
    ax.axvline(x, color='gray', alpha=0.2, linewidth=1.5, linestyle='--', zorder=0)

# Add horizontal grid
ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.8, zorder=0)

# Title and labels
ax.set_title('Integrated Performance Metrics and Sentiment Analysis\nParallel Coordinates Visualization',
            fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Normalized Scale', fontsize=12, fontweight='bold', labelpad=10)

# Legend with custom styling
legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
                   fancybox=True, shadow=True, framealpha=0.95)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')

# Add actual values as text annotations
for idx, row in company_data.iterrows():
    company = row['company']
    y_offset = 0.05 if company == 'Amazon UK' else -0.08
    
    for i, metric in enumerate(metrics):
        value = row[metric]
        normalized_val = normalized_data.iloc[idx][metric]
        
        # Format value based on metric type
        if 'pct' in metric or 'rate' in metric:
            text = f"{value:.1f}%"
        elif 'score' in metric:
            text = f"{value:.1f}"
        else:
            text = f"£{value:.2f}"
        
        ax.text(x_positions[i], normalized_val + y_offset, text,
               ha='center', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.8, edgecolor='gray', linewidth=0.5))

# Add background gradient effect
gradient = np.linspace(0, 1, 256).reshape(256, 1)
gradient = np.hstack((gradient, gradient))
ax.imshow(gradient, extent=[ax.get_xlim()[0], ax.get_xlim()[1], 
                            ax.get_ylim()[0], ax.get_ylim()[1]],
         aspect='auto', alpha=0.05, cmap='Blues', zorder=-1)

plt.tight_layout()
plt.savefig('graph7_parallel_coordinates_sentiment.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.show()

# Print summary
print("\nIntegrated Performance Metrics Summary:")
print("=" * 80)
for idx, row in company_data.iterrows():
    print(f"\n{row['company']}:")
    for metric, label in zip(metrics, metric_labels):
        label_clean = label.replace('\n', ' ')
        value = row[metric]
        if 'pct' in metric or 'rate' in metric:
            print(f"  {label_clean}: {value:.2f}%")
        elif 'score' in metric:
            print(f"  {label_clean}: {value:.2f}")
        else:
            print(f"  {label_clean}: £{value:.2f}")
