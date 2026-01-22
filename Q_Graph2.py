import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# File paths
quantitative = r"C:\Users\aravi\Downloads\Survey_Data-20260122T230150Z-1-001\Survey_Data\Quantitative Analysis_data_20251230_131750.xlsx"

# Load data
logistics_df = pd.read_excel(quantitative, sheet_name='Logistics')
environmental_df = pd.read_excel(quantitative, sheet_name='Environmental')

# Merge datasets
merge_keys = ['city']
merged_df = pd.merge(logistics_df, environmental_df, on=merge_keys, how='inner')

# Define target and features
target_column = 'actual_delivery_time_min'
feature_columns = [
    'total_stops',
    'distance_km',
    'vehicle_utilization_pct',
    'packages_delivered',
    'traffic_flow_vehicles_per_hour',
    'avg_speed_kmh',
    'temperature_celsius',
    'urban_density_pop_per_sqkm'
]

# Filter available features
available_features = [col for col in feature_columns if col in merged_df.columns]

# Prepare data
merged_df = merged_df.dropna(subset=[target_column])
X = merged_df[available_features].fillna(merged_df[available_features].mean())
y = merged_df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest to get feature importance
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

# Get top 5 features
top_5 = feature_importance.tail(5)

# Create gradient colors (blue gradient with blur effect)
colors = [
    f'rgba(54, 162, 235, {0.4 + i*0.15})' 
    for i in range(len(top_5))
]

# Feature name mapping for better readability
feature_labels = {
    'total_stops': 'Total Stops',
    'distance_km': 'Distance (km)',
    'vehicle_utilization_pct': 'Vehicle Utilization (%)',
    'packages_delivered': 'Packages Delivered',
    'traffic_flow_vehicles_per_hour': 'Traffic Flow (vehicles/hr)',
    'avg_speed_kmh': 'Average Speed (km/h)',
    'temperature_celsius': 'Temperature (°C)',
    'urban_density_pop_per_sqkm': 'Urban Density (pop/km²)'
}

readable_labels = [feature_labels.get(f, f) for f in top_5['feature']]

# Create horizontal bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    y=readable_labels,
    x=top_5['importance'],
    orientation='h',
    marker=dict(
        color=colors,
        line=dict(color='rgba(255, 255, 255, 0.6)', width=2)
    ),
    text=[f'{val:.4f}' for val in top_5['importance']],
    textposition='outside',
    hovertemplate='&lt;b&gt;%{y}&lt;/b&gt;&lt;br&gt;Importance: %{x:.4f}&lt;extra&gt;&lt;/extra&gt;'
))

# Update layout with gradient background
fig.update_layout(
    title={
        'text': 'Top 5 Feature Importance Rankings&lt;br&gt;&lt;sub&gt;Random Forest Model Analysis&lt;/sub&gt;',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': '#1a1a1a'}
    },
    xaxis=dict(
        title='Importance Score',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)'
    ),
    yaxis=dict(
        title='Features',
        title_font=dict(size=14),
        tickfont=dict(size=12)
    ),
    plot_bgcolor='rgba(240, 248, 255, 0.3)',  # Light blue blur effect
    paper_bgcolor='white',
    font=dict(size=12, family='Arial, sans-serif'),
    height=500,
    width=900,
    margin=dict(l=200, r=100, t=100, b=80),
    hovermode='y unified'
)

# Save and display
fig.write_html('graph2_feature_importance.html')
print("\nTop 5 Feature Importance Rankings:")
for idx, row in top_5.iterrows():
    print(f"{feature_labels.get(row['feature'], row['feature'])}: {row['importance']:.4f}")

fig.show()
