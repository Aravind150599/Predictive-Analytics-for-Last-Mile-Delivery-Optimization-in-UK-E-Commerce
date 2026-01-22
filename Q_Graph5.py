import pandas as pd
import plotly.graph_objects as go
import numpy as np

# File path
quantitative = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"

# Load data
logistics_df = pd.read_excel(quantitative, sheet_name='Logistics')

# Calculate first-attempt success rate using packages_delivered and failed_deliveries
# First-attempt success = packages_delivered / (packages_delivered + failed_deliveries)
logistics_df['first_attempt_success_rate'] = (
    logistics_df['packages_delivered'] / 
    (logistics_df['packages_delivered'] + logistics_df['failed_deliveries'])
)

# Calculate first-attempt success rates by city
city_success = logistics_df.groupby('city').agg({
    'first_attempt_success_rate': ['mean', 'std', 'count']
}).reset_index()

# Flatten column names
city_success.columns = ['city', 'success_rate', 'std_dev', 'count']

# Convert to percentage
city_success['success_rate'] = city_success['success_rate'] * 100
city_success['std_dev'] = city_success['std_dev'] * 100

# Calculate standard error for error bars
city_success['std_error'] = city_success['std_dev'] / np.sqrt(city_success['count'])

# Sort by success rate for better visualization
city_success = city_success.sort_values('success_rate', ascending=False)

# Create figure
fig = go.Figure()

# Add bar chart with error bars
fig.add_trace(go.Bar(
    x=city_success['city'],
    y=city_success['success_rate'],
    error_y=dict(
        type='data',
        array=city_success['std_error'],
        visible=True,
        color='rgba(0, 0, 0, 0.5)',
        thickness=1.5,
        width=4
    ),
    marker=dict(
        color=city_success['success_rate'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title='Success Rate (%)',
            ticksuffix='%'
        ),
        line=dict(color='rgba(0, 0, 0, 0.3)', width=1)
    ),
    text=[f'{val:.1f}%' for val in city_success['success_rate']],
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>Success Rate: %{y:.2f}%<br>Std Error: ±%{error_y.array:.2f}%<extra></extra>'
))

# Update layout
fig.update_layout(
    title={
        'text': 'First-Attempt Success Rates by City<br><sub>With Standard Error Bars</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': '#1a1a1a'}
    },
    xaxis=dict(
        title='City',
        title_font=dict(size=14),
        tickfont=dict(size=11),
        tickangle=-45
    ),
    yaxis=dict(
        title='First-Attempt Success Rate (%)',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
        range=[0, 100]
    ),
    plot_bgcolor='rgba(240, 248, 255, 0.3)',
    paper_bgcolor='white',
    font=dict(size=12, family='Arial, sans-serif'),
    height=600,
    width=1200,
    margin=dict(l=80, r=80, t=120, b=120),
    hovermode='closest'
)

# Save and display
fig.write_html('graph5_first_attempt_success_by_city.html')

# Print summary statistics
print("\n=== First-Attempt Success Rates by City ===\n")
for _, row in city_success.iterrows():
    print(f"{row['city']}:")
    print(f"  Success Rate: {row['success_rate']:.2f}%")
    print(f"  Std Error: ±{row['std_error']:.2f}%")
    print(f"  Sample Size: {int(row['count'])}\n")

fig.show()