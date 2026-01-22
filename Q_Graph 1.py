import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# File paths
quantitative = r"C:\Users\aravi\Downloads\Survey_Data-20260122T230150Z-1-001\Survey_Data\Quantitative Analysis_data_20251230_131750.xlsx"
qualitative_file = r"C:\Users\aravi\Downloads\Survey_Data-20260122T230150Z-1-001\Survey_Data\Qualitative Analysis_data_20251230_132246.xlsx"

# Load data
logistics_df = pd.read_excel(quantitative, sheet_name='Logistics')
environmental_df = pd.read_excel(quantitative, sheet_name='Environmental')

# Print columns to debug
print("Logistics columns:", logistics_df.columns.tolist())
print("Environmental columns:", environmental_df.columns.tolist())

# Find common columns for merge
common_cols = list(set(logistics_df.columns) & set(environmental_df.columns))
print("Common columns:", common_cols)

# Use city as merge key
merge_keys = ['city']
print(f"Using merge keys: {merge_keys}")

# Merge datasets
merged_df = pd.merge(logistics_df, environmental_df, on=merge_keys, how='inner')
print(f"Merged dataset size: {len(merged_df)} rows")
print("Merged columns:", merged_df.columns.tolist())

# Use actual_delivery_time_min as the target column
target_column = 'actual_delivery_time_min'
print(f"Using target column: {target_column}")

# Define features and target
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

# Filter to only use columns that exist
available_features = [col for col in feature_columns if col in merged_df.columns]
print(f"Available features: {available_features}")

if len(available_features) == 0:
    raise ValueError("No feature columns found in the merged dataset")

# Prepare data - remove rows with missing target values
merged_df = merged_df.dropna(subset=[target_column])
X = merged_df[available_features].fillna(merged_df[available_features].mean())
y = merged_df[target_column]

print(f"Dataset size after removing missing values: {len(X)} rows")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

# Data for visualization
models = ['Random Forest', 'Linear Regression']
rmse_values = [rf_rmse, lr_rmse]
r2_values = [rf_r2, lr_r2]

# Create figure
fig = go.Figure()

colors_rmse = ['rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)']
colors_r2 = ['rgba(255, 159, 64, 0.8)', 'rgba(75, 192, 192, 0.8)']

# Add RMSE bars
fig.add_trace(go.Bar(
    name='RMSE (minutes)',
    x=models,
    y=rmse_values,
    marker=dict(color=colors_rmse, line=dict(color='white', width=2)),
    text=[f'{val:.2f}' for val in rmse_values],
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>RMSE: %{y:.2f} minutes<extra></extra>',
    yaxis='y',
    offsetgroup=1
))

# Add R² bars
fig.add_trace(go.Bar(
    name='R² Score',
    x=models,
    y=r2_values,
    marker=dict(color=colors_r2, line=dict(color='white', width=2)),
    text=[f'{val:.4f}' for val in r2_values],
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>',
    yaxis='y2',
    offsetgroup=2
))

# Update layout
fig.update_layout(
    title={
        'text': 'Predictive Model Performance Comparison<br><sub>Random Forest vs Linear Regression Models</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': '#1a1a1a'}
    },
    xaxis=dict(
        title='Model Type',
        title_font=dict(size=14),
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title='RMSE (minutes)',
        title_font=dict(size=14, color='rgba(255, 99, 132, 1)'),
        tickfont=dict(size=12, color='rgba(255, 99, 132, 1)'),
        side='left'
    ),
    yaxis2=dict(
        title='R² Score',
        title_font=dict(size=14, color='rgba(75, 192, 192, 1)'),
        tickfont=dict(size=12, color='rgba(75, 192, 192, 1)'),
        overlaying='y',
        side='right'
    ),
    barmode='group',
    bargap=0.3,
    bargroupgap=0.1,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12, family='Arial, sans-serif'),
    height=600,
    width=900,
    margin=dict(l=80, r=80, t=100, b=80),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        font=dict(size=12)
    ),
    hovermode='x unified'
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(200, 200, 200, 0.3)'
)

# Save and display
fig.write_html('graph1_model_comparison.html')
print(f"\nRandom Forest - RMSE: {rf_rmse:.2f} min, R²: {rf_r2:.4f}")
print(f"Linear Regression - RMSE: {lr_rmse:.2f} min, R²: {lr_r2:.4f}")

fig.show()
