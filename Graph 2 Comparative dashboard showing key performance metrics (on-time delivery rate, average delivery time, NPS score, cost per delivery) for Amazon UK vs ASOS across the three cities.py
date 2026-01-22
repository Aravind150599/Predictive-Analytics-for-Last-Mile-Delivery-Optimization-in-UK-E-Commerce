import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = r"C:\Users\aravi\Downloads\Survey_Data-20260122T230150Z-1-001\Survey_Data\Quantitative Analysis_data_20251230_131750.xlsx"
performance_df = pd.read_excel(file_path, sheet_name='Performance')

# Filter for the three cities
cities = ['London', 'Manchester', 'Birmingham']
performance_df = performance_df[performance_df['city'].isin(cities)]

# Select key metrics
metrics = [
    ('on_time_delivery_rate_pct', 'On-Time Delivery Rate (%)'),
    ('avg_delivery_time_min', 'Average Delivery Time (min)'),
    ('nps_score', 'NPS Score'),
    ('cost_per_delivery_gbp', 'Cost per Delivery (£)')
]

# Create a 2x2 subplot dashboard with increased spacing
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Comparative Performance Metrics Amazon UK vs ASOS', fontsize=18, fontweight='bold', y=0.995)

# Add more space between subplots
plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.96, bottom=0.05, left=0.08, right=0.95)

# Flatten axes for easier iteration
axes = axes.flatten()

# Blue gradient colors for the two companies
colors = ['#0066CC', '#66B3FF']  # Darker blue for Amazon UK, lighter blue for ASOS

# Create line charts for each metric
for idx, (metric_col, metric_label) in enumerate(metrics)
    ax = axes[idx]
    
    # Group by company and city
    grouped_data = performance_df.groupby(['company', 'city'])[metric_col].mean().reset_index()
    
    # Plot lines for each company
    for company_idx, company in enumerate(['Amazon UK', 'ASOS'])
        company_data = grouped_data[grouped_data['company'] == company]
        company_data = company_data.set_index('city').reindex(['London', 'Manchester', 'Birmingham'])
        
        ax.plot(company_data.index, company_data[metric_col], 
                marker='o', linewidth=2.5, markersize=10,
                color=colors[company_idx], label=company)
        
        # Add value labels on points with better positioning
        for x, y in zip(company_data.index, company_data[metric_col])
            ax.annotate(f'{y.1f}', (x, y), textcoords=offset points, 
                       xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    # Set title with padding
    ax.set_title(metric_label, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('City', fontsize=12, labelpad=8)
    ax.set_ylabel(metric_label, fontsize=12, labelpad=8)
    
    # Position legend to avoid overlap
    ax.legend(title='Company', fontsize=11, title_fontsize=11, loc='best', framealpha=0.9)
    
    # Improve grid visibility
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Set x-axis with proper spacing
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels(cities, fontsize=11)
    
    # Adjust y-axis to prevent label overlap
    ax.tick_params(axis='y', labelsize=10)
    
    # Add margins to prevent clipping
    ax.margins(x=0.1, y=0.15)

plt.savefig('graph2_comparative_dashboard_lines.png', dpi=300, bbox_inches='tight')

plt.show()
