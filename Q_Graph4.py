import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# File path
quantitative = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"

# Load data
logistics_df = pd.read_excel(quantitative, sheet_name='Logistics')

# Filter data for Amazon UK and ASOS
amazon_data = logistics_df[logistics_df['company'] == 'Amazon UK']
asos_data = logistics_df[logistics_df['company'] == 'ASOS']

# Calculate metrics for both companies
def calculate_metrics(data):
    metrics = {}
    
    # Delivery Time
    if 'actual_delivery_time_min' in data.columns:
        metrics['avg_delivery_time'] = data['actual_delivery_time_min'].mean()
    
    # On-Time Rate
    if 'estimated_delivery_time_min' in data.columns and 'actual_delivery_time_min' in data.columns:
        metrics['on_time_rate'] = (data['actual_delivery_time_min'] <= data['estimated_delivery_time_min']).mean() * 100
    
    # Customer Satisfaction
    if 'customer_satisfaction_score' in data.columns:
        metrics['customer_satisfaction'] = data['customer_satisfaction_score'].mean()
    
    # NPS (if available, otherwise calculate from satisfaction)
    if 'nps_score' in data.columns:
        metrics['nps'] = data['nps_score'].mean()
    elif 'customer_satisfaction_score' in data.columns:
        # Estimate NPS from satisfaction (scaled appropriately)
        metrics['nps'] = (data['customer_satisfaction_score'].mean() - 3) * 25
    
    return metrics

amazon_metrics = calculate_metrics(amazon_data)
asos_metrics = calculate_metrics(asos_data)

# Determine which metrics are available
available_metrics = set(amazon_metrics.keys()) & set(asos_metrics.keys())

# Create subplot configuration based on available metrics
metric_info = {
    'avg_delivery_time': ('Average Delivery Time (minutes)', 'min'),
    'on_time_rate': ('On-Time Delivery Rate (%)', '%'),
    'customer_satisfaction': ('Customer Satisfaction Score', ''),
    'nps': ('Net Promoter Score (NPS)', '')
}

# Filter to only available metrics
subplot_titles = [metric_info[m][0] for m in ['avg_delivery_time', 'on_time_rate', 'customer_satisfaction', 'nps'] if m in available_metrics]

# Create subplots
num_metrics = len(available_metrics)
if num_metrics >= 4:
    rows, cols = 2, 2
elif num_metrics == 3:
    rows, cols = 2, 2
elif num_metrics == 2:
    rows, cols = 1, 2
else:
    rows, cols = 1, 1

fig = make_subplots(
    rows=rows, cols=cols,
    subplot_titles=subplot_titles,
    specs=[[{'type': 'bar'}] * cols for _ in range(rows)],
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

# Color scheme with gradients
amazon_color = 'rgba(54, 162, 235, 0.7)'
amazon_line = 'rgba(54, 162, 235, 1)'
asos_color = 'rgba(255, 159, 64, 0.7)'
asos_line = 'rgba(255, 159, 64, 1)'

companies = ['Amazon UK', 'ASOS']

# Add traces for each available metric
metric_order = ['avg_delivery_time', 'on_time_rate', 'customer_satisfaction', 'nps']
subplot_idx = 0

for metric in metric_order:
    if metric in available_metrics:
        subplot_idx += 1
        row = (subplot_idx - 1) // cols + 1
        col = (subplot_idx - 1) % cols + 1
        
        values = [amazon_metrics[metric], asos_metrics[metric]]
        unit = metric_info[metric][1]
        
        fig.add_trace(go.Bar(
            x=companies,
            y=values,
            marker=dict(
                color=[amazon_color, asos_color],
                line=dict(color=[amazon_line, asos_line], width=2)
            ),
            text=[f'{val:.1f}{unit}' if unit else f'{val:.2f}' for val in values],
            textposition='outside',
            showlegend=False,
            hovertemplate='%{x}: %{y:.2f}' + unit + '<extra></extra>'
        ), row=row, col=col)

# Update layout
fig.update_layout(
    title={
        'text': 'Multi-Metric Comparison Dashboard<br><sub>Amazon UK vs ASOS Performance Analysis</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 22, 'family': 'Arial, sans-serif', 'color': '#1a1a1a'}
    },
    plot_bgcolor='rgba(240, 248, 255, 0.3)',
    paper_bgcolor='white',
    font=dict(size=11, family='Arial, sans-serif'),
    height=800,
    width=1200,
    margin=dict(l=80, r=80, t=120, b=60),
    hovermode='closest'
)

# Update axes with grid
for i in range(1, rows + 1):
    for j in range(1, cols + 1):
        fig.update_xaxes(
            tickfont=dict(size=11),
            showgrid=False,
            row=i, col=j
        )
        fig.update_yaxes(
            tickfont=dict(size=10),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.3)',
            row=i, col=j
        )

# Save and display
fig.write_html('graph4_multi_metric_dashboard.html')

# Print summary
print("\n=== Multi-Metric Comparison Dashboard ===\n")
print("Amazon UK Metrics:")
for key, value in amazon_metrics.items():
    print(f"  {key}: {value:.2f}")
print("\nASOS Metrics:")
for key, value in asos_metrics.items():
    print(f"  {key}: {value:.2f}")

fig.show()
