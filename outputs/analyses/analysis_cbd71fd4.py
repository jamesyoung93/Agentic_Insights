# Research Question: What are the key attributes of customers who exhibit high spending behavior in the coffee shop chain business?

# Load necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load data
customers = pd.read_csv('data/customers.csv')
transactions = pd.read_csv('data/transactions.csv')

# Merge data
merged_data = pd.merge(customers, transactions, on='customer_id', how='inner')

# Filter high spending customers
high_spending_customers = merged_data[merged_data['amount'] > merged_data['amount'].mean()]

# Statistical analysis
mean_age = high_spending_customers['age'].mean()
total_spending = high_spending_customers['amount'].sum()
correlation_income_spending = high_spending_customers['income'].corr(high_spending_customers['amount'])

# Save plots
plt.figure(figsize=(8, 6))
plt.scatter(high_spending_customers['age'], high_spending_customers['amount'])
plt.xlabel('Age')
plt.ylabel('Amount Spent')
plt.title('Age vs. Amount Spent for High Spending Customers')
plt.savefig('outputs/analyses/high_spending_age_amount.png')
plt.close()

# Summary dictionary
summary = {
    'mean_age_high_spending_customers': mean_age,
    'total_spending_high_spending_customers': total_spending,
    'correlation_income_spending_high_spending_customers': correlation_income_spending
}

summary