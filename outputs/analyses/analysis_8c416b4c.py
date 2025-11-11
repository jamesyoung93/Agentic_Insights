# Research Question: Is there a significant difference in customer satisfaction levels between loyalty program members and non-members?

# Load necessary libraries
import pandas as pd
from scipy.stats import ttest_ind

# Load data
customers = pd.read_csv('data/customers.csv')
transactions = pd.read_csv('data/transactions.csv')

# Merge data on customer_id
merged_data = pd.merge(customers, transactions, on='customer_id', how='inner')

# Perform t-test for customer satisfaction levels between loyalty program members and non-members
loyalty_members = merged_data[merged_data['loyalty_member'] == 1]['satisfaction_score']
non_loyalty_members = merged_data[merged_data['loyalty_member'] == 0]['satisfaction_score']

t_stat, p_val = ttest_ind(loyalty_members, non_loyalty_members, equal_var=False)

# Calculate means
loyalty_mean = loyalty_members.mean()
non_loyalty_mean = non_loyalty_members.mean()

# Print key findings
print("Key Findings:")
print(f"Mean satisfaction score for loyalty program members: {loyalty_mean}")
print(f"Mean satisfaction score for non-members: {non_loyalty_mean}")
print(f"P-value from t-test: {p_val}")

# Save plot of satisfaction scores
import matplotlib.pyplot as plt
plt.hist(loyalty_members, alpha=0.5, label='Loyalty Members', color='blue')
plt.hist(non_loyalty_members, alpha=0.5, label='Non-Members', color='orange')
plt.xlabel('Satisfaction Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('outputs/analyses/satisfaction_scores.png')

# Return summary dictionary
summary_dict = {
    'loyalty_mean': loyalty_mean,
    'non_loyalty_mean': non_loyalty_mean,
    'p_value': p_val
}
summary_dict