"""
Analysis Approaches Reference
Common analytical methods and code patterns for customer analytics
"""

# ============================================
# 1. DESCRIPTIVE STATISTICS & EXPLORATION
# ============================================

APPROACH_DESCRIPTIVE = """
# Load and explore data
import pandas as pd
import numpy as np

# Basic statistics
df.describe()
df.info()
df.groupby('category')['metric'].agg(['mean', 'median', 'std', 'count'])

# Check for missing values
df.isnull().sum()

# Distribution analysis
from scipy import stats
stats.normaltest(df['metric'])  # Test for normality
"""

# ============================================
# 2. CUSTOMER SEGMENTATION
# ============================================

APPROACH_CLUSTERING = """
# K-means clustering for customer segmentation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Prepare features
features = ['age', 'income', 'visit_frequency', 'avg_spending']
X = df[features].dropna()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using elbow method
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Fit final model
kmeans = KMeans(n_clusters=4, random_state=42)
df['segment'] = kmeans.fit_predict(X_scaled)

# Profile segments
df.groupby('segment')[features + ['satisfaction']].mean()
"""

# ============================================
# 3. TIME SERIES ANALYSIS
# ============================================

APPROACH_TIME_SERIES = """
# Time series decomposition and trend analysis
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
ts = df.set_index('date')['revenue'].resample('W').sum()

# Decompose
decomposition = seasonal_decompose(ts, model='additive', period=52)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Test for stationarity
adf_test = adfuller(ts.dropna())
print(f'ADF Statistic: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')

# Calculate growth rates
ts_pct_change = ts.pct_change()
"""

# ============================================
# 4. COHORT ANALYSIS
# ============================================

APPROACH_COHORT = """
# Customer cohort retention analysis
import pandas as pd
import numpy as np

# Create cohort based on signup month
df['cohort'] = pd.to_datetime(df['signup_date']).dt.to_period('M')
df['order_period'] = pd.to_datetime(df['transaction_date']).dt.to_period('M')

# Calculate cohort age
df['cohort_age'] = (df['order_period'] - df['cohort']).apply(lambda x: x.n)

# Cohort size
cohort_size = df.groupby('cohort')['customer_id'].nunique()

# Retention matrix
retention = df.groupby(['cohort', 'cohort_age'])['customer_id'].nunique()
retention = retention.unstack(fill_value=0)
retention_pct = retention.divide(cohort_size, axis=0) * 100
"""

# ============================================
# 5. CORRELATION & CAUSATION
# ============================================

APPROACH_CORRELATION = """
# Correlation analysis and significance testing
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Pearson correlation matrix
corr_matrix = df[numeric_cols].corr()

# Individual correlation with p-value
corr, pval = pearsonr(df['variable1'], df['variable2'])
print(f"Correlation: {corr:.3f}, p-value: {pval:.3f}")

# Spearman for non-linear relationships
corr_spearman, pval = spearmanr(df['variable1'], df['variable2'])

# Partial correlation (controlling for confounders)
from pingouin import partial_corr
partial_corr(data=df, x='var1', y='var2', covar='confounder')
"""

# ============================================
# 6. REGRESSION ANALYSIS
# ============================================

APPROACH_REGRESSION = """
# Linear regression for predictive modeling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm

# Simple linear regression with statsmodels (includes p-values)
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']
X = sm.add_constant(X)  # Add intercept

model = sm.OLS(y, X).fit()
print(model.summary())  # Detailed statistics

# sklearn for prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
"""

# ============================================
# 7. A/B TESTING & HYPOTHESIS TESTING
# ============================================

APPROACH_HYPOTHESIS_TEST = """
# Statistical hypothesis testing
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

# T-test for comparing two groups (parametric)
group_a = df[df['group'] == 'A']['metric']
group_b = df[df['group'] == 'B']['metric']
t_stat, p_value = ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

# Mann-Whitney U test (non-parametric)
u_stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')

# Chi-square test for categorical variables
contingency_table = pd.crosstab(df['category1'], df['category2'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square: {chi2:.3f}, p-value: {p_value:.3f}")
"""

# ============================================
# 8. CUSTOMER LIFETIME VALUE (CLV)
# ============================================

APPROACH_CLV = """
# Calculate Customer Lifetime Value
import pandas as pd
import numpy as np

# Historical CLV
clv_df = df.groupby('customer_id').agg({
    'amount': 'sum',  # Total revenue
    'transaction_id': 'count',  # Purchase frequency
    'date': ['min', 'max']  # Tenure
})

clv_df.columns = ['total_revenue', 'purchase_count', 'first_purchase', 'last_purchase']
clv_df['tenure_days'] = (clv_df['last_purchase'] - clv_df['first_purchase']).dt.days
clv_df['avg_order_value'] = clv_df['total_revenue'] / clv_df['purchase_count']

# Predictive CLV (simplified BG/NBD approach)
# Average order value * Purchase frequency * Expected lifetime
clv_df['predicted_lifetime_months'] = 24  # Assumption
clv_df['purchases_per_month'] = clv_df['purchase_count'] / (clv_df['tenure_days'] / 30)
clv_df['predicted_clv'] = (clv_df['avg_order_value'] * 
                           clv_df['purchases_per_month'] * 
                           clv_df['predicted_lifetime_months'])
"""

# ============================================
# 9. CHURN PREDICTION
# ============================================

APPROACH_CHURN = """
# Binary classification for churn prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Define churn (e.g., no purchase in last 90 days)
latest_date = df['date'].max()
last_purchase = df.groupby('customer_id')['date'].max()
df['days_since_last_purchase'] = (latest_date - last_purchase).dt.days
df['churned'] = (df['days_since_last_purchase'] > 90).astype(int)

# Features
features = ['age', 'income', 'avg_purchase_value', 'purchase_frequency', 
            'loyalty_member', 'mobile_app_user', 'competitor_visits']
X = df[features]
y = df['churned']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
"""

# ============================================
# 10. GEOGRAPHIC ANALYSIS
# ============================================

APPROACH_GEOGRAPHIC = """
# Geographic segmentation and comparison
import pandas as pd
from scipy.stats import f_oneway

# Compare metrics across regions
regional_comparison = df.groupby('region').agg({
    'amount': ['mean', 'median', 'std'],
    'satisfaction_score': ['mean', 'std'],
    'visit_frequency': 'mean',
    'customer_id': 'count'
}).round(2)

# ANOVA to test if differences are significant
regions = df['region'].unique()
region_data = [df[df['region'] == r]['amount'] for r in regions]
f_stat, p_value = f_oneway(*region_data)
print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")

# City-level analysis
city_metrics = df.groupby(['region', 'city']).agg({
    'amount': 'mean',
    'satisfaction_score': 'mean',
    'customer_id': 'nunique'
}).sort_values(('amount', 'mean'), ascending=False)
"""

# Analysis approaches registry
APPROACHES = {
    'descriptive': APPROACH_DESCRIPTIVE,
    'clustering': APPROACH_CLUSTERING,
    'time_series': APPROACH_TIME_SERIES,
    'cohort': APPROACH_COHORT,
    'correlation': APPROACH_CORRELATION,
    'regression': APPROACH_REGRESSION,
    'hypothesis_testing': APPROACH_HYPOTHESIS_TEST,
    'clv': APPROACH_CLV,
    'churn': APPROACH_CHURN,
    'geographic': APPROACH_GEOGRAPHIC
}

# Approach selection guide
APPROACH_GUIDE = """
WHEN TO USE EACH APPROACH:

1. DESCRIPTIVE - Always start here for initial exploration
2. CLUSTERING - For customer segmentation questions
3. TIME_SERIES - For questions about trends, seasonality, forecasting
4. COHORT - For retention analysis and customer lifecycle questions
5. CORRELATION - To find relationships between variables
6. REGRESSION - For prediction and understanding feature importance
7. HYPOTHESIS_TESTING - To test specific claims (e.g., A vs B comparison)
8. CLV - To understand customer value and prioritize segments
9. CHURN - To predict and prevent customer attrition
10. GEOGRAPHIC - To analyze regional/city differences
"""
