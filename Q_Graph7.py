import pandas as pd
import matplotlib.pyplot as plt

# Load the data
quantitative = r"C:\Users\Kishore\Downloads\Kathirvelan\Quantitative Analysis_data_20251230_131750.xlsx"
qualitative_file = r"C:\Users\Kishore\Downloads\Kathirvelan\Qualitative Analysis_data_20251230_132246.xlsx"

# Load the qualitative data (Interviews sheet)
df = pd.read_excel(qualitative_file, sheet_name='Interviews')

# Aggregate sentiment by company
sentiment_counts = df.groupby(['company', 'implementation_sentiment']).size().unstack(fill_value=0)

# Create line graph
fig, ax = plt.subplots(figsize=(12, 8))

companies = sentiment_counts.index
sentiments = sentiment_counts.columns

# Plot a line for each sentiment
for sentiment in sentiments:
    ax.plot(companies, sentiment_counts[sentiment], marker='o', linewidth=2, 
            markersize=8, label=sentiment)
    
    # Add value labels
    for i, value in enumerate(sentiment_counts[sentiment]):
        ax.text(i, value, str(value), ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Company', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Implementation Sentiment by Company', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(title='Sentiment', loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()