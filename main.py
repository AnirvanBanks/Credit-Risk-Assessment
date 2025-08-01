print("Credit Risk Assessment Using Probability and Statistics\n")

print("Problem Statement:Develop a credit risk assessment framework that uses historical loan data and probabilistic/statistical analysis. The goal is to identify and evaluate the risk of loan default and dynamically update this risk using Bayesian inference.\n")

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 01. Data Collection
# Path to your downloaded LendingClub dataset
input_path = '/Users/home/Desktop/Python VSCode/project/01_data_collection/LendingClub.csv'

# Full local paths
input_path = '/Users/home/Desktop/Python VSCode/project/LendingClub.csv'
output_path = '/Users/home/Desktop/Python VSCode/project/cleaned_loan_data.csv'

# Load sampled LendingClub data
print("🔹 Loading sampled data...")
df = pd.read_csv(input_path, low_memory=False)
print("Loaded. Initial shape:", df.shape)

# Drop irrelevant or high-cardinality columns
drop_cols = [
    'id', 'member_id', 'emp_title', 'url', 'desc', 'title', 'zip_code',
    'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d',
    'issue_d', 'application_type', 'policy_code', 'addr_state'
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Drop rows with loan_status = 'Current' or other non-final statuses
if 'loan_status' in df.columns:
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    print("🧹 Filtered to only Fully Paid and Charged Off loans.")

# Drop rows with missing key features
required = ['loan_amnt', 'term', 'int_rate', 'annual_inc']
df.dropna(subset=required, inplace=True)

# Fill missing annual income with median (if any)
if df['annual_inc'].isnull().sum() > 0:
    df['annual_inc'].fillna(df['annual_inc'].median(), inplace=True)

# Clean and convert 'term' to numeric
df['term'] = df['term'].str.extract(r'(\d+)').astype(float)

# Encode target variable: 1 = Defaulted, 0 = Fully Paid
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
df.rename(columns={'loan_status': 'default'}, inplace=True)

# One-hot encode remaining object columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Final shape and save
print("Final shape after cleaning:", df.shape)
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to:\n{output_path}")

# 📊 03: Exploratory Data Analysis (EDA)

df = pd.read_csv("/Users/home/Desktop/Python VSCode/project/cleaned_loan_data.csv")

# Basic EDA
print(df.describe())
print(df.info())

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Default rates
if 'loan_status_Fully Paid' in df.columns:
    print("\nDefault Rate:")
    print(1 - df['loan_status_Fully Paid'].mean())

# 📈 04: Probability Distributions

df = pd.read_csv("/Users/home/Desktop/Python VSCode/project/cleaned_loan_data.csv")

# Example: Fit normal distribution to annual income
if 'annual_inc' in df.columns:
    data = df['annual_inc']
    mu, sigma = stats.norm.fit(data)

    plt.hist(data, bins=50, density=True, alpha=0.6, color='skyblue')
    x = np.linspace(min(data), max(data), 1000)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r')
    plt.title(f'Normal Fit: mu={mu:.2f}, sigma={sigma:.2f}')
    plt.show()

# 🧮 05: Risk Profiling

df = pd.read_csv("/Users/home/Desktop/Python VSCode/project/cleaned_loan_data.csv")

# Sample risk factors
risk_factors = ['loan_amnt', 'int_rate', 'annual_inc', 'term']

# Create a simple risk score (weighted example, normalize values)
for col in risk_factors:
    df[f'norm_{col}'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Weight risk factors arbitrarily
df['risk_score'] = (0.3 * df['norm_loan_amnt'] +
                    0.3 * df['norm_int_rate'] +
                    0.2 * (1 - df['norm_annual_inc']) +
                    0.2 * df['norm_term'])

# Create buckets: Low, Medium, High Risk
df['risk_bucket'] = pd.qcut(df['risk_score'], q=3, labels=["Low", "Medium", "High"])

df[['loan_amnt', 'int_rate', 'annual_inc', 'risk_score', 'risk_bucket']].head(10)
df.to_csv('risk_profiled_data.csv', index=False)
print("Risk profiling completed and saved to 'risk_profiled_data.csv'")

# 🧠 06: Bayesian Updating

# Load profiled data
df = pd.read_csv('/Users/home/Desktop/Python VSCode/project/risk_profiled_data.csv')

# Example:
# P(Default) = 0.2
# P(High Risk | Default) = 0.6
# P(High Risk) = 0.3 (proportion in data)

prior = 0.2
likelihood = 0.6
evidence = (df['risk_bucket'] == 'High').mean()

posterior = (likelihood * prior) / evidence

print(f"Updated P(Default | High Risk): {posterior:.2%}")

# 🤖 07: ML Baseline Model

df = pd.read_csv("/Users/home/Desktop/Python VSCode/project/cleaned_loan_data.csv")

# Ensure target is binary
df['default'] = df['loan_status_Fully Paid'].apply(lambda x: 0 if x == 1 else 1)

X = df.drop(columns=['loan_status_Fully Paid', 'default'])
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds))
print("ROC AUC Score:", roc_auc_score(y_test, probs))