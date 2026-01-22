import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load the data
performance_file = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"
qualitative_file = r"C:\Users\Kishore\Downloads\Kathirvelan\Qualitative Analysis_data_20251230_132246.xlsx"

# Load relevant sheets
performance_df = pd.read_excel(performance_file, sheet_name='Performance')
interviews_df = pd.read_excel(qualitative_file, sheet_name='Interviews')

# Aggregate data by company
perf_metrics = performance_df.groupby('company').agg({
    'on_time_delivery_rate_pct': 'mean',
    'customer_satisfaction_avg': 'mean',
    'nps_score': 'mean',
    'cost_per_delivery_gbp': 'mean',
    'first_attempt_success_rate_pct': 'mean'
}).reset_index()

# Get implementation factors from interviews
impl_factors = interviews_df.groupby(['company', 'implementation_sentiment']).size().reset_index(name='count')

# Define nodes for Sankey diagram
# Qualitative factors (source nodes)
qual_nodes = [
    'Very Positive Implementation',
    'Positive Implementation', 
    'Neutral Implementation',
    'Concerned Implementation',
    'Critical Implementation'
]

# Quantitative outcomes (target nodes)
quant_nodes = [
    'High On-Time Delivery (>87%)',
    'High Customer Satisfaction (>4.2)',
    'Strong NPS Score (>49)',
    'Low Cost per Delivery (<£9)',
    'High First-Attempt Success (>80%)'
]

# Company nodes
company_nodes = ['Amazon UK', 'ASOS']

# Create complete node list
all_nodes = qual_nodes + company_nodes + quant_nodes

# Map sentiment to node indices
sentiment_map = {
    'Very Positive': 0,
    'Positive': 1,
    'Neutral': 2,
    'Concerned': 3,
    'Critical': 4
}

# Initialize source, target, value, and color lists
sources = []
targets = []
values = []
colors = []

# Map qualitative factors to companies based on interview data
for _, row in impl_factors.iterrows():
    company = row['company']
    sentiment = row['implementation_sentiment']
    count = row['count']
    
    if sentiment in sentiment_map:
        source_idx = sentiment_map[sentiment]
        company_idx = len(qual_nodes) + company_nodes.index(company)
        
        sources.append(source_idx)
        targets.append(company_idx)
        values.append(count)
        
        # Color coding with transparency for blur effect
        if sentiment == 'Very Positive':
            colors.append('rgba(0, 180, 0, 0.4)')
        elif sentiment == 'Positive':
            colors.append('rgba(100, 200, 100, 0.4)')
        elif sentiment == 'Neutral':
            colors.append('rgba(150, 150, 150, 0.4)')
        elif sentiment == 'Concerned':
            colors.append('rgba(255, 180, 0, 0.4)')
        else:  # Critical
            colors.append('rgba(255, 0, 0, 0.4)')

# Map companies to quantitative outcomes
for idx, row in perf_metrics.iterrows():
    company = row['company']
    company_idx = len(qual_nodes) + company_nodes.index(company)
    
    # On-time delivery
    if row['on_time_delivery_rate_pct'] > 87:
        sources.append(company_idx)
        targets.append(len(qual_nodes) + len(company_nodes) + 0)
        values.append(row['on_time_delivery_rate_pct'] / 10)
        colors.append('rgba(0, 102, 204, 0.5)' if company == 'Amazon UK' else 'rgba(255, 107, 53, 0.5)')
    
    # Customer satisfaction
    if row['customer_satisfaction_avg'] > 4.2:
        sources.append(company_idx)
        targets.append(len(qual_nodes) + len(company_nodes) + 1)
        values.append(row['customer_satisfaction_avg'] * 20)
        colors.append('rgba(0, 102, 204, 0.5)' if company == 'Amazon UK' else 'rgba(255, 107, 53, 0.5)')
    
    # NPS score
    if row['nps_score'] > 49:
        sources.append(company_idx)
        targets.append(len(qual_nodes) + len(company_nodes) + 2)
        values.append(row['nps_score'] / 5)
        colors.append('rgba(0, 102, 204, 0.5)' if company == 'Amazon UK' else 'rgba(255, 107, 53, 0.5)')
    
    # Cost per delivery
    if row['cost_per_delivery_gbp'] < 9:
        sources.append(company_idx)
        targets.append(len(qual_nodes) + len(company_nodes) + 3)
        values.append(10 - row['cost_per_delivery_gbp'])
        colors.append('rgba(0, 102, 204, 0.5)' if company == 'Amazon UK' else 'rgba(255, 107, 53, 0.5)')
    
    # First-attempt success
    if row['first_attempt_success_rate_pct'] > 80:
        sources.append(company_idx)
        targets.append(len(qual_nodes) + len(company_nodes) + 4)
        values.append(row['first_attempt_success_rate_pct'] / 10)
        colors.append('rgba(0, 102, 204, 0.5)' if company == 'Amazon UK' else 'rgba(255, 107, 53, 0.5)')

# Create node colors
node_colors = []
for node in all_nodes:
    if 'Very Positive' in node:
        node_colors.append('rgba(0, 180, 0, 0.8)')
    elif 'Positive' in node:
        node_colors.append('rgba(100, 200, 100, 0.8)')
    elif 'Neutral' in node:
        node_colors.append('rgba(150, 150, 150, 0.8)')
    elif 'Concerned' in node:
        node_colors.append('rgba(255, 180, 0, 0.8)')
    elif 'Critical' in node:
        node_colors.append('rgba(255, 0, 0, 0.8)')
    elif node == 'Amazon UK':
        node_colors.append('rgba(0, 102, 204, 0.9)')
    elif node == 'ASOS':
        node_colors.append('rgba(255, 107, 53, 0.9)')
    else:
        node_colors.append('rgba(0, 60, 120, 0.8)')

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=30,
        line=dict(color='white', width=2),
        label=all_nodes,
        color=node_colors,
        customdata=[f"<b>{node}</b>" for node in all_nodes],
        hovertemplate='%{customdata}<br>%{value:.1f}<extra></extra>'
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=colors,
        hovertemplate='Flow: %{value:.1f}<extra></extra>'
    )
)])

# Update layout
fig.update_layout(
    title={
        'text': 'Matrix Visualization - Qualitative Implementation Factors to Quantitative Outcomes<br><sub>Connection strength indicated by flow thickness with blur gradient effect</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'family': 'Arial, sans-serif', 'color': '#1a1a1a'}
    },
    font=dict(size=11, family='Arial, sans-serif'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=800,
    margin=dict(l=20, r=20, t=100, b=20)
)

# Save the HTML file (this will always work)
fig.write_html('graph8_sankey_matrix.html')

print("\nGraph 8 created successfully!")
print(f"\nSankey Diagram Summary:")
print(f"Total Flows: {len(sources)}")
print(f"Qualitative Factors (Source): {len(qual_nodes)}")
print(f"Companies (Intermediate): {len(company_nodes)}")
print(f"Quantitative Outcomes (Target): {len(quant_nodes)}")
print(f"\nInterview Data Summary:")
print(impl_factors)
print(f"\nFile saved:")
print("- graph8_sankey_matrix.html (interactive)")
print("\nTo create a PNG file, please upgrade your packages:")
print("  pip install -U plotly kaleido")

# Display the figure
fig.show()
