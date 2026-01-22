import pandas as pd
import plotly.graph_objects as go
import numpy as np

# File path
quantitative = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"

# Load data
logistics_df = pd.read_excel(quantitative, sheet_name='Logistics')

# Filter data for Amazon UK and ASOS
amazon_data = logistics_df[logistics_df['company'] == 'Amazon UK']
asos_data = logistics_df[logistics_df['company'] == 'ASOS']

# Calculate performance metrics for both companies
metrics = {
    'Amazon UK': {
        'Avg Delivery Time (min)': amazon_data['actual_delivery_time_min'].mean(),
        'Vehicle Utilization (%)': amazon_data['vehicle_utilization_pct'].mean(),
        'Avg Distance (km)': amazon_data['distance_km'].mean()
    },
    'ASOS': {
        'Avg Delivery Time (min)': asos_data['actual_delivery_time_min'].mean(),
        'Vehicle Utilization (%)': asos_data['vehicle_utilization_pct'].mean(),
        'Avg Distance (km)': asos_data['distance_km'].mean()
    }
}

# Add on-time rate if columns exist to calculate it
if 'estimated_delivery_time_min' in logistics_df.columns and 'actual_delivery_time_min' in logistics_df.columns:
    amazon_on_time = (amazon_data['actual_delivery_time_min'] <= amazon_data['estimated_delivery_time_min']).mean() * 100
    asos_on_time = (asos_data['actual_delivery_time_min'] <= asos_data['estimated_delivery_time_min']).mean() * 100
    metrics['Amazon UK']['On-Time Rate (%)'] = amazon_on_time
    metrics['ASOS']['On-Time Rate (%)'] = asos_on_time

# Add customer satisfaction if it exists
if 'customer_satisfaction_score' in logistics_df.columns:
    metrics['Amazon UK']['Customer Satisfaction'] = amazon_data['customer_satisfaction_score'].mean()
    metrics['ASOS']['Customer Satisfaction'] = asos_data['customer_satisfaction_score'].mean()

# Add packages delivered if it exists
if 'packages_delivered' in logistics_df.columns:
    metrics['Amazon UK']['Avg Packages Delivered'] = amazon_data['packages_delivered'].mean()
    metrics['ASOS']['Avg Packages Delivered'] = asos_data['packages_delivered'].mean()

# Remove None values
for company in metrics:
    metrics[company] = {k: v for k, v in metrics[company].items() if v is not None and not pd.isna(v)}

# Prepare data for visualization
metric_names = list(metrics['Amazon UK'].keys())
amazon_values = list(metrics['Amazon UK'].values())
asos_values = list(metrics['ASOS'].values())

# Create figure
fig = go.Figure()

# Amazon UK bars with blue gradient
fig.add_trace(go.Bar(
    name='Amazon UK',
    x=metric_names,
    y=amazon_values,
    marker=dict(
        color=['rgba(54, 162, 235, 0.7)' for _ in range(len(metric_names))],
        line=dict(color='rgba(54, 162, 235, 1)', width=2),
        pattern=dict(shape='/', bgcolor='rgba(54, 162, 235, 0.3)', fgcolor='rgba(54, 162, 235, 0.5)')
    ),
    text=[f'{val:.2f}' for val in amazon_values],
    textposition='outside',
    hovertemplate='<b>Amazon UK</b><br>%{x}: %{y:.2f}<extra></extra>'
))

# ASOS bars with orange gradient
fig.add_trace(go.Bar(
    name='ASOS',
    x=metric_names,
    y=asos_values,
    marker=dict(
        color=['rgba(255, 159, 64, 0.7)' for _ in range(len(metric_names))],
        line=dict(color='rgba(255, 159, 64, 1)', width=2),
        pattern=dict(shape='\\', bgcolor='rgba(255, 159, 64, 0.3)', fgcolor='rgba(255, 159, 64, 0.5)')
    ),
    text=[f'{val:.2f}' for val in asos_values],
    textposition='outside',
    hovertemplate='<b>ASOS</b><br>%{x}: %{y:.2f}<extra></extra>'
))

# Update layout with blur gradient background
fig.update_layout(
    title={
        'text': 'Amazon UK vs ASOS Model Performance Comparison<br><sub>Side-by-Side Analysis of Key Metrics</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': '#1a1a1a'}
    },
    xaxis=dict(
        title='Performance Metrics',
        title_font=dict(size=14),
        tickfont=dict(size=11),
        tickangle=-30
    ),
    yaxis=dict(
        title='Value',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)'
    ),
    barmode='group',
    bargap=0.2,
    bargroupgap=0.1,
    plot_bgcolor='rgba(240, 248, 255, 0.3)',
    paper_bgcolor='white',
    font=dict(size=12, family='Arial, sans-serif'),
    height=600,
    width=1000,
    margin=dict(l=80, r=80, t=120, b=120),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        font=dict(size=13),
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(200, 200, 200, 0.5)',
        borderwidth=1
    ),
    hovermode='x unified'
)

# Save and display
fig.write_html('graph3_amazon_vs_asos_comparison.html')

# Print summary statistics
print("\n=== Amazon UK vs ASOS Performance Comparison ===\n")
for metric in metric_names:
    amazon_val = metrics['Amazon UK'][metric]
    asos_val = metrics['ASOS'][metric]
    diff = amazon_val - asos_val
    print(f"{metric}:")
    print(f"  Amazon UK: {amazon_val:.2f}")
    print(f"  ASOS: {asos_val:.2f}")
    print(f"  Difference: {diff:.2f}\n")

fig.show()