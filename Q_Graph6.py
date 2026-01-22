import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the quantitative data
quantitative = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"
df = pd.read_excel(quantitative, sheet_name='Performance')

# Create the bubble chart
plt.figure(figsize=(12, 8))

# Create separate series for each company
companies = df['company'].unique()
colors = ['#1f77b4', '#ff7f0e']  # Blue and orange

for i, company in enumerate(companies):
    company_data = df[df['company'] == company]
    
    plt.scatter(
        x=company_data['on_time_delivery_rate_pct'],
        y=company_data['cost_per_delivery_gbp'],
        s=company_data['avg_delivery_time_min'] * 10,  # Scale bubble size
        alpha=0.6,
        c=colors[i],
        label=company,
        edgecolors='black',
        linewidth=0.5
    )

plt.xlabel('On-Time Delivery Rate (%)', fontsize=12, fontweight='bold')
plt.ylabel('Cost per Delivery (GBP)', fontsize=12, fontweight='bold')
plt.title('On-Time Delivery Rate vs. Cost per Delivery\n(Bubble size represents delivery time)', 
          fontsize=14, fontweight='bold', pad=20)
plt.legend(title='Company', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

# Add a size legend
bubble_sizes = [20, 40, 60]
bubble_labels = ['20 min', '40 min', '60 min']
legend_bubbles = []
for size in bubble_sizes:
    legend_bubbles.append(plt.scatter([], [], s=size*10, c='gray', alpha=0.6, edgecolors='black'))

plt.legend(legend_bubbles, bubble_labels, scatterpoints=1, title='Delivery Time',
          loc='upper right', frameon=True, fontsize=9)
plt.legend(companies, loc='upper left', title='Company')

plt.tight_layout()
plt.show()
