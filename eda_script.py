import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory
os.makedirs("notebooks/plots", exist_ok=True)

# Load data
df = pd.read_csv("data/customer_churn_data.csv")

# Create SQLite connection and load data
conn = sqlite3.connect(":memory:")
df.to_sql("customers", conn, index=False, if_exists="replace")

def run_query(query, title):
    print(f"\n--- {title} ---")
    result = pd.read_sql_query(query, conn)
    print(result)
    return result

# SQL Queries
# 1. Churn Rate by Contract Type
query1 = """
SELECT Contract, 
       COUNT(*) as Total_Customers,
       SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as Churned_Customers,
       ROUND(CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) as Churn_Rate
FROM customers
GROUP BY Contract
ORDER BY Churn_Rate DESC;
"""
run_query(query1, "Churn Rate by Contract Type")

# 2. Average Monthly Charges by Internet Service and Churn Status
query2 = """
SELECT InternetService, Churn,
       ROUND(AVG(MonthlyCharges), 2) as Avg_Monthly_Charges,
       COUNT(*) as Count
FROM customers
GROUP BY InternetService, Churn
ORDER BY InternetService, Churn;
"""
run_query(query2, "Avg Monthly Charges by Internet Service & Churn")

# 3. Churn by Tenure Groups
query3 = """
SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-12 Months'
        WHEN tenure <= 24 THEN '13-24 Months'
        WHEN tenure <= 48 THEN '25-48 Months'
        ELSE '49+ Months'
    END as Tenure_Group,
    COUNT(*) as Total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as Churned,
    ROUND(CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) as Churn_Rate
FROM customers
GROUP BY Tenure_Group
ORDER BY Churn_Rate DESC;
"""
run_query(query3, "Churn by Tenure Group")

# Visualizations

# 1. Churn Distribution
plt.figure(figsize=(6, 6))
df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('Distribution of Churn')
plt.ylabel('')
plt.savefig("notebooks/plots/churn_distribution.png")
plt.close()

# 2. Correlation Matrix (Numerical Features)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
# Convert Churn to numeric for correlation
df['Churn_Num'] = df['Churn'].map({'Yes': 1, 'No': 0})
numeric_df_corr = pd.concat([numeric_df, df['Churn_Num']], axis=1)

sns.heatmap(numeric_df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig("notebooks/plots/correlation_matrix.png")
plt.close()

# 3. Churn by Contract Type (Bar Chart)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn by Contract Type')
plt.savefig("notebooks/plots/churn_by_contract.png")
plt.close()

print("\nEDA completed. Plots saved to notebooks/plots/")
