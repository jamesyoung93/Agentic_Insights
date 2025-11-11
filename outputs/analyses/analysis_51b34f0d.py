# Research Question: What is the impact of loyalty program membership duration on customer satisfaction and retention rates?

# Load necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Load data
customers = pd.read_csv('data/customers.csv')
transactions = pd.read_csv('data/transactions.csv')

# Merge data on customer_id
merged_data = pd.merge(customers, transactions, on='customer_id')

# Filter data for loyalty program members
loyalty_members = merged_data[merged_data['loyalty_member'] == 1]

# Filter data for non-loyalty program members
non_loyalty_members = merged_data[merged_data['loyalty_member'] == 0]

# T-test for satisfaction scores between loyalty members and non-members
t_stat, p_val = ttest_ind(loyalty_members['satisfaction_score'], non_loyalty_members['satisfaction_score'], equal_var=False)

# Calculate retention rates for loyalty program members and non-members
retention_rate_loyalty = loyalty_members['customer_id'].nunique() / customers[customers['loyalty_member'] == 1]['customer_id'].nunique()
retention_rate_non_loyalty = non_loyalty_members['customer_id'].nunique() / customers[customers['loyalty_member'] == 0]['customer_id'].nunique()

# Summary dictionary with findings
summary = {
    'loyalty_members_mean_satisfaction': loyalty_members['satisfaction_score'].mean(),
    'non_loyalty_members_mean_satisfaction': non_loyalty_members['satisfaction_score'].mean(),
    'p_value_satisfaction_ttest': p_val,
    'retention_rate_loyalty': retention_rate_loyalty,
    'retention_rate_non_loyalty': retention_rate_non_loyalty
}

# Save plots to outputs/analyses/ (if any)

summary