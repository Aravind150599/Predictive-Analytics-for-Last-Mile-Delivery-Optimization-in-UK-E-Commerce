import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

# Load the data
file_path = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"
logistics_df = pd.read_excel(file_path, sheet_name='Logistics')

# Filter for Amazon UK and ASOS
companies = ['Amazon UK', 'ASOS']
logistics_df = logistics_df[logistics_df['company'].isin(companies)]

# Remove any rows with missing values in the required columns
logistics_df = logistics_df.dropna(subset=['predicted_delivery_time_min', 'actual_delivery_time_min'])

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Predicted vs Actual Delivery Times: Amazon UK vs ASOS', 
             fontsize=16, fontweight='bold', y=0.98)

plt.subplots_adjust(left=0.08, right=0.94, top=0.90, bottom=0.12, wspace=0.25)

# Define colors for gradient effect
colors_dict = {
    'Amazon UK': '#FF9900',  # Amazon orange
    'ASOS': '#0770CF'  # ASOS blue
}

for idx, company in enumerate(companies):
    ax = axes[idx]
    company_data = logistics_df[logistics_df['company'] == company].copy()
    
    x = company_data['predicted_delivery_time_min'].values
    y = company_data['actual_delivery_time_min'].values
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    
    # Calculate confidence interval
    predict_error = y - (slope * x + intercept)
    residual_std = np.std(predict_error)
    ci = 1.96 * residual_std  # 95% confidence interval
    
    # Create blur effect by plotting multiple layers
    base_color = colors_dict[company]
    
    # Plot blur layers (outermost to innermost)
    for blur_idx in range(5, 0, -1):
        alpha = 0.05 * blur_idx
        size = 100 + (blur_idx * 20)
        ax.scatter(x, y, s=size, c=base_color, alpha=alpha, edgecolors='none')
    
    # Plot main scatter points
    ax.scatter(x, y, s=100, c=base_color, alpha=0.6, edgecolors='#333333', 
               linewidth=1.5, label='Actual Data', zorder=3)
    
    # Plot regression line
    ax.plot(line_x, line_y, color='#333333', linewidth=2.5, 
            label=f'Regression Line (R²={r_value**2:.3f})', zorder=4)
    
    # Plot confidence interval with gradient fill
    for i in range(10):
        alpha_ci = 0.15 - (i * 0.012)
        ci_factor = 1 - (i * 0.1)
        ax.fill_between(line_x, line_y - ci*ci_factor, line_y + ci*ci_factor, 
                        color=base_color, alpha=alpha_ci, zorder=1)
    
    # Plot perfect prediction line (y=x)
    perfect_line = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 100)
    ax.plot(perfect_line, perfect_line, 'k--', linewidth=1.5, alpha=0.5, 
            label='Perfect Prediction', zorder=2)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y - (slope * x + intercept))**2))
    r2 = r2_score(y, slope * x + intercept)
    
    # Add statistics text box
    stats_text = f'RMSE: {rmse:.2f} min\nR²: {r2:.4f}\ny = {slope:.3f}x + {intercept:.2f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     alpha=0.8, edgecolor='gray'))
    
    # Styling
    ax.set_title(f'{company}', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Delivery Time (minutes)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual Delivery Time (minutes)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_axisbelow(True)
    
    # Make axes equal for better comparison
    max_val = max(x.max(), y.max())
    min_val = min(x.min(), y.min())
    ax.set_xlim(min_val - 5, max_val + 5)
    ax.set_ylim(min_val - 5, max_val + 5)

plt.savefig('graph5_predicted_vs_actual_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nDelivery Time Prediction Performance Summary:")
print("="*60)
for company in companies:
    company_data = logistics_df[logistics_df['company'] == company]
    x = company_data['predicted_delivery_time_min'].values
    y = company_data['actual_delivery_time_min'].values
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    rmse = np.sqrt(np.mean((y - (slope * x + intercept))**2))
    r2 = r2_score(y, slope * x + intercept)
    print(f"\n{company}:")
    print(f"  Sample size: {len(x)}")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  R²: {r2:.4f}")
    print(f"  Regression equation: y = {slope:.3f}x + {intercept:.2f}")
    print(f"  Mean predicted time: {x.mean():.2f} min")
    print(f"  Mean actual time: {y.mean():.2f} min")