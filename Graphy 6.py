import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load data
data_path = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"
perf_data = pd.read_excel(data_path, sheet_name='Performance')

# Define metrics to visualize
metric_list = [
    ('on_time_delivery_rate_pct', 'On-Time Delivery (%)'),
    ('avg_delivery_time_min', 'Avg Delivery Time (min)'),
    ('customer_satisfaction_avg', 'Customer Satisfaction'),
    ('cost_per_delivery_gbp', 'Cost per Delivery (£)'),
    ('carbon_emissions_kg', 'Carbon Emissions (kg)'),
    ('first_attempt_success_rate_pct', 'First Attempt Success (%)')
]

# Setup figure
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Performance Metrics: Amazon UK vs ASOS by City', 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.06, wspace=0.25, hspace=0.30)

# Color scheme
company_colors = {'Amazon UK': '#FF9900', 'ASOS': '#0770CF'}
city_list = ['London', 'Manchester', 'Birmingham']

for i, (metric_name, metric_title) in enumerate(metric_list):
    r = i // 3
    c = i % 3
    axis = axs[r, c]
    
    # Collect data for plotting
    plot_data = []
    x_labels = []
    x_pos = []
    colors_list = []
    
    position = 0
    for comp in ['Amazon UK', 'ASOS']:
        for city in city_list:
            subset = perf_data[
                (perf_data['company'] == comp) & 
                (perf_data['city'] == city)
            ][metric_name].dropna()
            
            if len(subset) > 0:
                plot_data.append(subset.values)
                x_labels.append(city[:3])
                x_pos.append(position)
                colors_list.append(company_colors[comp])
                position += 1
        position += 1
    
    # Create boxplots
    box_plot = axis.boxplot(plot_data, positions=x_pos, widths=0.6,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(color='black', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='gray', 
                                          markersize=4, alpha=0.5))
    
    # Apply colors with gradient blur effect
    for box, col in zip(box_plot['boxes'], colors_list):
        box.set_facecolor(col)
        box.set_alpha(0.7)
        box.set_edgecolor('black')
        
        # Blur effect layers
        verts = box.get_path().vertices
        for b in range(5, 0, -1):
            blur_alpha = 0.05 * b
            axis.add_patch(plt.Polygon(verts, facecolor=col, 
                                      alpha=blur_alpha, edgecolor='none', 
                                      transform=axis.transData, zorder=0))
    
    # Add trend lines
    for comp_idx, comp in enumerate(['Amazon UK', 'ASOS']):
        trend_pos = []
        trend_vals = []
        
        start_idx = comp_idx * (len(city_list) + 1)
        end_idx = start_idx + len(city_list)
        
        for idx in range(start_idx, end_idx):
            if idx < len(plot_data):
                trend_pos.append(x_pos[idx])
                trend_vals.append(np.mean(plot_data[idx]))
        
        if len(trend_pos) > 1:
            axis.plot(trend_pos, trend_vals, 
                     color='black', linewidth=2.5, 
                     marker='o', markersize=8, alpha=0.8,
                     label=f'{comp} Trend', zorder=5)
            
            # Blur effect for trend line
            for b in range(3):
                blur_alpha = 0.1 - (b * 0.03)
                axis.plot(trend_pos, trend_vals, 
                         color='black', linewidth=3 + b,
                         alpha=blur_alpha, zorder=4)
    
    # Format subplot
    axis.set_title(metric_title, fontsize=11, fontweight='bold', pad=10)
    axis.set_ylabel('Value', fontsize=9, fontweight='bold')
    axis.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    axis.set_axisbelow(True)
    axis.set_xticks(x_pos)
    axis.set_xticklabels(x_labels, fontsize=8)
    
    # Company labels
    amazon_positions = x_pos[:len(city_list)]
    asos_positions = x_pos[len(city_list):len(city_list)*2]
    
    if len(amazon_positions) > 0:
        amazon_mid = np.mean(amazon_positions)
        axis.text(amazon_mid, axis.get_ylim()[0], 'Amazon UK', 
                  ha='center', va='top', fontsize=9, fontweight='bold',
                  color='black', transform=axis.get_xaxis_transform(),
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           alpha=0.8, edgecolor='black'))
    
    if len(asos_positions) > 0:
        asos_mid = np.mean(asos_positions)
        axis.text(asos_mid, axis.get_ylim()[0], 'ASOS', 
                  ha='center', va='top', fontsize=9, fontweight='bold',
                  color='black', transform=axis.get_xaxis_transform(),
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           alpha=0.8, edgecolor='black'))
    
    if i == 0:
        axis.legend(loc='upper right', fontsize=8, framealpha=0.9)

plt.savefig('graph6_performance_comparison_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("\nPerformance Metrics Summary by Company and City:")
print("="*80)
for metric_name, metric_title in metric_list:
    print(f"\n{metric_title}:")
    for comp in ['Amazon UK', 'ASOS']:
        print(f"\n  {comp}:")
        for city in city_list:
            subset = perf_data[
                (perf_data['company'] == comp) & 
                (perf_data['city'] == city)
            ][metric_name].dropna()
            if len(subset) > 0:
                print(f"    {city}: Mean={subset.mean():.2f}, "
                      f"Median={subset.median():.2f}, "
                      f"Std={subset.std():.2f}")