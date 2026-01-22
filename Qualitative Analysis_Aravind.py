import pandas as pd
import numpy as np
from collections import Counter
import re

# Load the data
file_path = r"C:\Users\Kishore\Downloads\Kathirvelan\Qualitative Analysis_data_20251230_132246.xlsx"

# Read all sheets
interviews_df = pd.read_excel(file_path, sheet_name='Interviews')
case_studies_df = pd.read_excel(file_path, sheet_name='Case Studies')
field_obs_df = pd.read_excel(file_path, sheet_name='Field Observations')
logistics_df = pd.read_excel(file_path, sheet_name='Logistics')
performance_df = pd.read_excel(file_path, sheet_name='Performance')

print("="*60)
print("QUALITATIVE ANALYSIS")
print("="*60)

# ============================================================
# 1. THEMATIC ANALYSIS: Code interview transcripts
# ============================================================
print("\n1. THEMATIC ANALYSIS - Interview Insights")
print("-"*60)

# Analyze key challenges mentioned
if 'key_challenges_mentioned' in interviews_df.columns:
    all_challenges = []
    for challenges in interviews_df['key_challenges_mentioned'].dropna():
        if isinstance(challenges, str):
            all_challenges.extend([c.strip() for c in challenges.split(',')])
    
    challenge_counts = Counter(all_challenges)
    print("\nTop Implementation Barriers:")
    for challenge, count in challenge_counts.most_common(10):
        print(f"  • {challenge}: {count} mentions")

# Analyze success factors
if 'success_factors_identified' in interviews_df.columns:
    all_success = []
    for factors in interviews_df['success_factors_identified'].dropna():
        if isinstance(factors, str):
            all_success.extend([f.strip() for f in factors.split(',')])
    
    success_counts = Counter(all_success)
    print("\nTop Success Factors:")
    for factor, count in success_counts.most_common(10):
        print(f"  • {factor}: {count} mentions")

# Sentiment analysis
if 'implementation_sentiment' in interviews_df.columns:
    sentiment_by_company = interviews_df.groupby('company')['implementation_sentiment'].value_counts()
    print("\nImplementation Sentiment by Company:")
    print(sentiment_by_company)

# ============================================================
# 2. CROSS-CASE COMPARISON: Amazon UK vs ASOS
# ============================================================
print("\n\n2. CROSS-CASE COMPARISON")
print("-"*60)

# Compare interview insights
print("\nInterview Comparison:")
interview_comparison = interviews_df.groupby('company').agg({
    'years_experience': 'mean',
    'interview_duration_min': 'mean',
    'transcript_length_words': 'mean'
}).round(2)
print(interview_comparison)

# Compare case study focus areas
if 'focus_area' in case_studies_df.columns:
    print("\nCase Study Focus Areas by Company:")
    focus_comparison = case_studies_df.groupby(['company', 'focus_area']).size().unstack(fill_value=0)
    print(focus_comparison)

# Compare field observations
print("\nField Observation Metrics:")
field_comparison = field_obs_df.groupby('city').agg({
    'deliveries_observed': 'sum',
    'delays_noted': 'sum',
    'successful_first_attempts': 'sum',
    'technology_issues_noted': 'sum',
    'parking_challenges': 'sum'
}).round(2)
print(field_comparison)

# Calculate success rates from observations
field_comparison['success_rate_%'] = (
    field_comparison['successful_first_attempts'] / 
    field_comparison['deliveries_observed'] * 100
).round(2)
print("\nFirst Attempt Success Rates by City:")
print(field_comparison[['success_rate_%']])

# ============================================================
# 3. TRIANGULATION: Validate quantitative with qualitative
# ============================================================
print("\n\n3. TRIANGULATION ANALYSIS")
print("-"*60)

# Link performance metrics with qualitative insights
performance_summary = performance_df.groupby('company').agg({
    'on_time_delivery_rate_pct': 'mean',
    'avg_delivery_time_min': 'mean',
    'customer_satisfaction_avg': 'mean',
    'nps_score': 'mean',
    'cost_per_delivery_gbp': 'mean'
}).round(2)

print("\nQuantitative Performance Metrics:")
print(performance_summary)

# Correlate with qualitative sentiment
if 'implementation_sentiment' in interviews_df.columns:
    sentiment_mapping = {'Positive': 3, 'Neutral': 2, 'Negative': 1}
    interviews_df['sentiment_score'] = interviews_df['implementation_sentiment'].map(sentiment_mapping)
    
    avg_sentiment = interviews_df.groupby('company')['sentiment_score'].mean().round(2)
    print("\nAverage Sentiment Score (1-3 scale):")
    print(avg_sentiment)

# Identify correlations
print("\n\nKEY FINDINGS:")
print("-"*60)

for company in performance_summary.index:
    perf = performance_summary.loc[company]
    print(f"\n{company}:")
    print(f"  • On-time delivery: {perf['on_time_delivery_rate_pct']:.1f}%")
    print(f"  • Avg delivery time: {perf['avg_delivery_time_min']:.1f} min")
    print(f"  • Customer satisfaction: {perf['customer_satisfaction_avg']:.2f}")
    print(f"  • NPS score: {perf['nps_score']:.1f}")
    print(f"  • Cost per delivery: £{perf['cost_per_delivery_gbp']:.2f}")
    
    # Add qualitative context
    company_interviews = interviews_df[interviews_df['company'] == company]
    if len(company_interviews) > 0:
        print(f"  • Interviews conducted: {len(company_interviews)}")
        if 'implementation_sentiment' in company_interviews.columns:
            pos_count = (company_interviews['implementation_sentiment'] == 'Positive').sum()
            print(f"  • Positive sentiment interviews: {pos_count}/{len(company_interviews)}")

# Export detailed findings
output_file = r"C:\Users\Kishore\Downloads\Kathirvelan\Qualitative_Analysis_Results.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    performance_summary.to_excel(writer, sheet_name='Performance Summary')
    interview_comparison.to_excel(writer, sheet_name='Interview Comparison')
    if 'focus_comparison' in locals():
        focus_comparison.to_excel(writer, sheet_name='Focus Areas')
    field_comparison.to_excel(writer, sheet_name='Field Observations')

print(f"\n\nDetailed results exported to: {output_file}")
