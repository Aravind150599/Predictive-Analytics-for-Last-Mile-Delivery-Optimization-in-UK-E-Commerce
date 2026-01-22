import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the data
file_path = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"
logistics_df = pd.read_excel(file_path, sheet_name='Logistics')

# Print available columns to verify
print("Available columns in the dataset:")
print(logistics_df.columns.tolist())

# Define potential features for prediction
potential_features = ['vehicle_utilization_pct', 'distance_km', 'total_stops', 
                      'packages_delivered', 'traffic_flow_vehicles_per_hour']

# Filter to only include columns that actually exist in the dataframe
feature_columns = [col for col in potential_features if col in logistics_df.columns]

print(f"\nUsing features: {feature_columns}")

# Check if we have the target column
if 'actual_delivery_time_min' not in logistics_df.columns:
    print("Error: 'actual_delivery_time_min' column not found!")
    print("Available columns are:", logistics_df.columns.tolist())
    exit()

# Prepare data - remove rows with missing values in relevant columns
data_clean = logistics_df[feature_columns + ['actual_delivery_time_min']].dropna()

print(f"\nTotal samples after cleaning: {len(data_clean)}")

X = data_clean[feature_columns]
y = data_clean['actual_delivery_time_min']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model to calculate feature importance
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Extract feature importance from the trained model
importance = rf_model.feature_importances_

# Create feature names for display (use actual column names)
feature_labels = []
for col in feature_columns:
    # Create readable labels
    label = col.replace('_', ' ').title()
    if 'pct' in col.lower():
        label = label.replace('Pct', '(%)')
    if 'km' in col.lower():
        label = label.replace('Km', '(km)')
    if 'gbp' in col.lower():
        label = label.replace('Gbp', '(£)')
    feature_labels.append(label)

# Sort features by importance
sorted_indices = np.argsort(importance)[::-1]
importance_sorted = importance[sorted_indices]
feature_labels_sorted = [feature_labels[i] for i in sorted_indices]

# Create the figure
fig, ax = plt.subplots(figsize=(12, 7))

# Create gradient colors (from light blue to dark blue)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_labels_sorted)))

# Create bars with gradient effect
bars = ax.barh(feature_labels_sorted, importance_sorted, color=colors, edgecolor='#003366', linewidth=2)

# Add blur effect by overlaying semi-transparent bars
for i, (bar, imp) in enumerate(zip(bars, importance_sorted)):
    # Create multiple layers with decreasing opacity for blur effect
    for j in range(5):
        alpha = 0.15 - (j * 0.02)
        offset = (j + 1) * 0.02
        ax.barh(i, imp, left=-offset*imp, height=0.8, 
                color=colors[i], alpha=alpha, edgecolor='none')

# Add value labels on bars
for i, (bar, imp) in enumerate(zip(bars, importance_sorted)):
    ax.text(imp + 0.005, i, f'{imp:.3f}', 
            va='center', fontweight='bold', fontsize=11)

# Styling
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Feature Importance for Delivery Time Predictions', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, max(importance_sorted) * 1.15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('graph4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Print feature importance summary
print("\nFeature Importance Summary (Calculated from Data):")
print("="*60)
for label, imp in zip(feature_labels_sorted, importance_sorted):
    print(f"{label}: {imp:.4f}")
print(f"\nTotal samples used: {len(data_clean)}")
print(f"Model R² score: {rf_model.score(X_test, y_test):.4f}")
