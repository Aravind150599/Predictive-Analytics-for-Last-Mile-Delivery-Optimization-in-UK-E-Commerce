import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Load data from Excel file
file_path = r"C:\Users\aravi\Downloads\Survey_Data-20260122T230150Z-1-001\Survey_Data\Quantitative Analysis_data_20251230_131750.xlsx"

def clean_columns(df):
    """Standardize column names to snake_case (lower_case_with_underscores)"""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

try:
    # Load all sheets
    logistics_df = clean_columns(pd.read_excel(file_path, sheet_name='Logistics'))
    customer_df = clean_columns(pd.read_excel(file_path, sheet_name='Customer'))
    environment_df = clean_columns(pd.read_excel(file_path, sheet_name='Environmental'))
    performance_df = clean_columns(pd.read_excel(file_path, sheet_name='Performance'))
    print("Data loaded and columns standardized successfully.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# ============================================
# 1. PREDICTIVE MODELING
# ============================================

print("=" * 60)
print("1. PREDICTIVE MODELING: Random Forest & Linear Regression")
print("=" * 60)

# --- STEP 1: ROBUST DATA PREPARATION ---

# Standardize Date formats
if 'date' in logistics_df.columns:
    logistics_df['date'] = pd.to_datetime(logistics_df['date'])
    logistics_df['join_date'] = logistics_df['date'].dt.normalize()
else:
    print("Error: 'date' column missing in Logistics sheet.")
    exit()

if 'timestamp' in environment_df.columns:
    environment_df['timestamp'] = pd.to_datetime(environment_df['timestamp'])
    environment_df['join_date'] = environment_df['timestamp'].dt.normalize()
elif 'date' in environment_df.columns:
    environment_df['date'] = pd.to_datetime(environment_df['date'])
    environment_df['join_date'] = environment_df['date'].dt.normalize()

# Standardize City names
logistics_df['city_clean'] = logistics_df['city'].astype(str).str.strip().str.lower()
environment_df['city_clean'] = environment_df['city'].astype(str).str.strip().str.lower()

print(f"Logistics rows: {len(logistics_df)}")
print(f"Environmental rows: {len(environment_df)}")

# Merge logistics with environment data
merged_df = logistics_df.merge(
    environment_df,
    left_on=['join_date', 'city_clean'],
    right_on=['join_date', 'city_clean'],
    how='left',
    suffixes=('', '_env')
)

print(f"Merged rows: {len(merged_df)}")

# --- STEP 2: HANDLE MISSING DATA (IMPUTATION) ---

# Numeric features list
numeric_feature_columns = [
    'total_stops', 'distance_km', 
    'vehicle_utilization_pct', 'packages_delivered',
    'traffic_flow_vehicles_per_hour', 'avg_speed_kmh',
    'temperature_celsius', 'urban_density_pop_per_sqkm'
]

# Fill missing numeric values
for col in numeric_feature_columns:
    if col in merged_df.columns:
        if merged_df[col].isnull().sum() > 0:
            print(f"Imputing missing values for {col} with mean...")
            merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    else:
        # If column is missing entirely, create it with 0s
        print(f"Warning: Column '{col}' missing. Creating with 0s.")
        merged_df[col] = 0

# --- STEP 3: ENCODING ---

# Handle traffic_density
if 'traffic_density' in merged_df.columns:
    traffic_density_map = {'low': 0, 'medium': 1, 'high': 2}
    merged_df['traffic_density'] = merged_df['traffic_density'].astype(str).str.lower()
    merged_df['traffic_density_encoded'] = merged_df['traffic_density'].map(traffic_density_map).fillna(1) # Default Medium
else:
    merged_df['traffic_density_encoded'] = 1

# Handle congestion_level
if 'congestion_level' in merged_df.columns:
    congestion_map = {'low': 0, 'medium': 1, 'high': 2}
    merged_df['congestion_level'] = merged_df['congestion_level'].astype(str).str.lower()
    merged_df['congestion_level_encoded'] = merged_df['congestion_level'].map(congestion_map).fillna(1)
    numeric_feature_columns.append('congestion_level_encoded')

# Encode vehicle type
if 'vehicle_type' in merged_df.columns:
    merged_df['vehicle_type'] = merged_df['vehicle_type'].fillna('Unknown')
    merged_df['vehicle_type_encoded'] = pd.Categorical(merged_df['vehicle_type']).codes
else:
    merged_df['vehicle_type_encoded'] = 0

# Encode weather
if 'weather_condition' in merged_df.columns:
    merged_df['weather_condition'] = merged_df['weather_condition'].fillna('Unknown')
    merged_df['weather_encoded'] = pd.Categorical(merged_df['weather_condition']).codes
else:
    merged_df['weather_encoded'] = 0

# Final Feature List
feature_columns = numeric_feature_columns + ['traffic_density_encoded', 'vehicle_type_encoded', 'weather_encoded']
target_col = 'actual_delivery_time_min'

# --- STEP 4: SAFE COLUMN SELECTION ---

# Identify which ID columns are actually present
id_cols = ['company']
if 'customer_id' in merged_df.columns:
    id_cols.append('customer_id')
elif 'order_id' in merged_df.columns:
    id_cols.append('order_id')
    print("Warning: 'customer_id' not found, using 'order_id' if available.")

# Select columns safely
available_cols = feature_columns + [target_col] + id_cols
# Filter to only columns that actually exist in merged_df
final_cols = [c for c in available_cols if c in merged_df.columns]

modeling_df = merged_df.dropna(subset=[target_col]).copy()
modeling_df = modeling_df[final_cols].copy()
modeling_df = modeling_df.fillna(0) 

print(f"Final dataset size for modeling: {len(modeling_df)}")

# --- STEP 5: MODELING ---

if len(modeling_df) < 5:
    print("\nERROR: Not enough data points to train models.")
else:
    X = modeling_df[feature_columns]
    y = modeling_df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    # Evaluate
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)

    print("\nRandom Forest Performance:")
    print(f"  RMSE: {rf_rmse:.2f} minutes")
    print(f"  R²: {rf_r2:.4f}")

    print("\nLinear Regression Performance:")
    print(f"  RMSE: {lr_rmse:.2f} minutes")
    print(f"  R²: {lr_r2:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())

    # ============================================
    # 2. COMPARATIVE ANALYSIS (Amazon UK vs ASOS)
    # ============================================
    
    if 'company' in modeling_df.columns:
        print("\n" + "=" * 60)
        print("2. COMPARATIVE ANALYSIS: Amazon UK vs ASOS")
        print("=" * 60)
        
        # Normalize company names just in case
        modeling_df['company'] = modeling_df['company'].astype(str).str.strip()
        
        companies = modeling_df['company'].unique()
        results = {}

        for company_name in companies:
            company_data = modeling_df[modeling_df['company'] == company_name]
            
            if len(company_data) < 10:
                continue
                
            X_company = company_data[feature_columns]
            y_company = company_data[target_col]
            
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                X_company, y_company, test_size=0.2, random_state=42
            )
            
            rf_company = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            rf_company.fit(X_train_c, y_train_c)
            rf_pred_c = rf_company.predict(X_test_c)
            
            results[company_name] = {
                'rmse': np.sqrt(mean_squared_error(y_test_c, rf_pred_c)),
                'r2': r2_score(y_test_c, rf_pred_c)
            }
            print(f"\n{company_name} Performance:")
            print(f"  RMSE: {results[company_name]['rmse']:.2f}")
            print(f"  R²: {results[company_name]['r2']:.4f}")

# ============================================
# 3. INTEGRATION ASSESSMENT (Safe Mode)
# ============================================

print("\n" + "=" * 60)
print("3. INTEGRATION ASSESSMENT")
print("=" * 60)

# Check if we can perform integration
if 'customer_id' not in logistics_df.columns:
    print("Skipping Integration Assessment: 'customer_id' column missing in Logistics sheet.")
    print("Cannot link Logistics data to Customer data without a common ID.")
else:
    # Merge customer data with logistics for integrated approach
    integrated_df = logistics_df.merge(
        customer_df,
        on=['customer_id', 'company'],
        how='inner',
        suffixes=('_logistics', '_customer')
    )

    if len(integrated_df) < 10:
        print("Insufficient data after merging Logistics and Customer sheets.")
    else:
        # Proceed with Integration Analysis...
        # Map categorical variables
        integrated_df['traffic_density_encoded'] = 1 # Default if missing
        if 'traffic_density' in integrated_df.columns:
             integrated_df['traffic_density_encoded'] = integrated_df['traffic_density'].map(
                 {'low':0, 'medium':1, 'high':2}).fillna(1)

        integrated_df['vehicle_type_encoded'] = pd.Categorical(integrated_df['vehicle_type']).codes

        # Define feature sets
        siloed_cols = ['total_stops', 'distance_km', 'vehicle_utilization_pct', 'packages_delivered', 'vehicle_type_encoded']
        integ_cols = siloed_cols + ['total_orders', 'avg_order_value_gbp', 'on_time_delivery_rate_pct', 'satisfaction_score']
        
        # Ensure all columns exist
        siloed_cols = [c for c in siloed_cols if c in integrated_df.columns]
        integ_cols = [c for c in integ_cols if c in integrated_df.columns]
        
        # Fill NaNs
        integrated_modeling = integrated_df.dropna(subset=[target_col]).copy()
        integrated_modeling = integrated_modeling.fillna(0)

        # Train/Test Split
        y_int = integrated_modeling[target_col]
        
        # Siloed Model
        X_silo = integrated_modeling[siloed_cols]
        X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_silo, y_int, test_size=0.2, random_state=42)
        rf_silo = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr_s, y_tr_s)
        pred_silo = rf_silo.predict(X_te_s)
        rmse_silo = np.sqrt(mean_squared_error(y_te_s, pred_silo))

        # Integrated Model
        X_integ = integrated_modeling[integ_cols]
        X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_integ, y_int, test_size=0.2, random_state=42)
        rf_integ = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr_i, y_tr_i)
        pred_integ = rf_integ.predict(X_te_i)
        rmse_integ = np.sqrt(mean_squared_error(y_te_i, pred_integ))

        print(f"\nSiloed RMSE: {rmse_silo:.2f}")
        print(f"Integrated RMSE: {rmse_integ:.2f}")
        print(f"Improvement: {rmse_silo - rmse_integ:.2f}")

# ============================================
# 4. PERFORMANCE METRICS
# ============================================
print("\n" + "=" * 60)
print("4. PERFORMANCE METRICS")
print("=" * 60)

# Safe aggregation
agg_dict = {
    'avg_delivery_time_min': 'mean',
    'on_time_delivery_rate_pct': 'mean',
    'nps_score': 'mean',
    'cost_per_delivery_gbp': 'mean'
}
# Only use columns that exist
agg_dict = {k: v for k, v in agg_dict.items() if k in performance_df.columns}

if not agg_dict:
    print("Performance columns missing.")
else:
    performance_summary = performance_df.groupby('company').agg(agg_dict).round(2)

    print(performance_summary)
