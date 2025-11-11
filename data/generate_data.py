"""
Generate simulated coffee shop customer data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

np.random.seed(42)

# Geography hierarchy
REGIONS = {
    'Northeast': ['New York', 'Boston', 'Philadelphia'],
    'South': ['Atlanta', 'Miami', 'Houston'],
    'Midwest': ['Chicago', 'Detroit', 'Minneapolis'],
    'West': ['Los Angeles', 'San Francisco', 'Seattle']
}

def generate_customers(n_customers=5000):
    """Generate customer master data"""
    customers = []
    customer_id = 1
    
    for region, cities in REGIONS.items():
        n_region = n_customers // len(REGIONS)
        for city in cities:
            n_city = n_region // len(cities)
            for _ in range(n_city):
                # Customer attributes
                age = np.random.normal(35, 12)
                income = np.random.lognormal(10.8, 0.5)
                
                customers.append({
                    'customer_id': customer_id,
                    'city': city,
                    'region': region,
                    'age': max(18, min(80, age)),
                    'income': max(20000, min(200000, income)),
                    'loyalty_member': np.random.choice([True, False], p=[0.6, 0.4]),
                    'signup_date': datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 365)),
                    'preferred_time': np.random.choice(['morning', 'afternoon', 'evening']),
                    'mobile_app_user': np.random.choice([True, False], p=[0.7, 0.3])
                })
                customer_id += 1
    
    return pd.DataFrame(customers)

def generate_transactions(customers_df, start_date='2023-01-01', end_date='2024-12-31'):
    """Generate transaction data with time series"""
    transactions = []
    transaction_id = 1
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days = (end - start).days
    
    for _, customer in customers_df.iterrows():
        # Customer visit frequency varies
        if customer['loyalty_member']:
            avg_visits_per_month = np.random.uniform(12, 25)
        else:
            avg_visits_per_month = np.random.uniform(2, 8)
        
        total_visits = int(avg_visits_per_month * 24)  # 24 months
        
        for _ in range(total_visits):
            # Random date in range
            visit_date = start + timedelta(days=np.random.randint(0, days))
            
            # Seasonal pattern (more visits in fall/winter)
            month = visit_date.month
            seasonal_boost = 1.2 if month in [10, 11, 12, 1, 2] else 1.0
            
            # Product mix
            product_types = ['coffee', 'espresso', 'latte', 'cappuccino', 'tea', 'pastry', 'sandwich']
            n_items = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            
            items = np.random.choice(product_types, size=n_items, replace=False)
            
            # Price varies by product
            price_map = {
                'coffee': (3.5, 0.5), 'espresso': (4.0, 0.5), 'latte': (5.5, 0.7),
                'cappuccino': (5.0, 0.6), 'tea': (3.0, 0.4), 'pastry': (4.5, 0.8),
                'sandwich': (8.0, 1.0)
            }
            
            total_amount = sum(np.random.normal(price_map[item][0], price_map[item][1]) 
                             for item in items)
            
            # Customer satisfaction (affected by various factors)
            base_satisfaction = 4.0
            if customer['mobile_app_user']:
                base_satisfaction += 0.3
            if customer['loyalty_member']:
                base_satisfaction += 0.2
            
            satisfaction = min(5.0, max(1.0, np.random.normal(base_satisfaction, 0.7)))
            
            # Competitor visit probability
            competitor_visit = np.random.choice([True, False], p=[0.3, 0.7])
            
            transactions.append({
                'transaction_id': transaction_id,
                'customer_id': customer['customer_id'],
                'date': visit_date.strftime('%Y-%m-%d'),
                'day_of_week': visit_date.strftime('%A'),
                'month': visit_date.month,
                'year': visit_date.year,
                'items': ','.join(items),
                'amount': round(total_amount, 2),
                'satisfaction_score': round(satisfaction, 1),
                'wait_time_minutes': max(1, np.random.normal(5, 2)),
                'used_mobile_app': customer['mobile_app_user'] and np.random.random() < 0.8,
                'promotion_used': np.random.choice([True, False], p=[0.15, 0.85]),
                'competitor_also_visited': competitor_visit
            })
            transaction_id += 1
    
    return pd.DataFrame(transactions)

def generate_competitor_data(customers_df):
    """Generate competitor interaction data"""
    competitor_data = []
    
    for _, customer in customers_df.iterrows():
        # Some customers visit competitors
        visits_competitor = np.random.choice([True, False], p=[0.4, 0.6])
        
        if visits_competitor:
            competitor_data.append({
                'customer_id': customer['customer_id'],
                'competitor_visits_per_month': np.random.uniform(1, 10),
                'competitor_satisfaction': np.random.uniform(2.5, 4.5),
                'price_sensitivity': np.random.uniform(0, 1),  # 0=not sensitive, 1=very sensitive
                'quality_preference': np.random.uniform(0, 1),  # 0=price focused, 1=quality focused
            })
    
    return pd.DataFrame(competitor_data)

if __name__ == '__main__':
    print("Generating customer data...")
    customers = generate_customers(5000)
    customers.to_csv('data/customers.csv', index=False)
    print(f"Created {len(customers)} customers")
    
    print("\nGenerating transaction data...")
    transactions = generate_transactions(customers)
    transactions.to_csv('data/transactions.csv', index=False)
    print(f"Created {len(transactions)} transactions")
    
    print("\nGenerating competitor data...")
    competitor = generate_competitor_data(customers)
    competitor.to_csv('data/competitor_data.csv', index=False)
    print(f"Created {len(competitor)} competitor records")
    
    # Generate data summary
    summary = {
        'customers': len(customers),
        'transactions': len(transactions),
        'date_range': {
            'start': transactions['date'].min(),
            'end': transactions['date'].max()
        },
        'regions': list(REGIONS.keys()),
        'cities': [city for cities in REGIONS.values() for city in cities],
        'avg_transaction_value': round(transactions['amount'].mean(), 2),
        'avg_satisfaction': round(transactions['satisfaction_score'].mean(), 2)
    }
    
    with open('data/data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Data generation complete!")
