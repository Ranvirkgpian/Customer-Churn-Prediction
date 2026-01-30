# Customer Churn Prediction & Strategy Report

## Executive Summary
This project aims to predict customer attrition and identify key drivers of churn using the Telco Customer Churn dataset. By leveraging machine learning and SQL analysis, we have identified high-risk segments and developed targeted strategies to improve customer retention.

## Key Findings (Data Analysis)

### 1. Contract Type is the #1 Predictor
Customers with **Month-to-month contracts** are significantly more likely to churn compared to those with one or two-year contracts.
*   **Month-to-month Churn Rate:** ~43%
*   **One-year Churn Rate:** ~11%
*   **Two-year Churn Rate:** ~3%

### 2. The "Danger Zone": First 12 Months
New customers are the most vulnerable. The churn rate is highest (~47%) during the first year of tenure. Once a customer stays beyond 12 months, their likelihood of leaving drops significantly. Long-term customers (>4 years) have a churn rate of only ~10%.

### 3. Internet Service Impact
Customers with **Fiber Optic** service show a surprisingly high propensity to churn (~42%) compared to DSL users (~19%). This suggests potential issues with pricing competitiveness or service reliability for the Fiber product, despite higher average charges.

## Model Performance
We trained three models: Logistic Regression, Random Forest, and Gradient Boosting.
*   **Best Model:** Logistic Regression
*   **Performance:** The model achieves an F1-Score of 0.64 and an Accuracy of 0.82. It effectively balances precision and recall to identify at-risk customers without excessive false alarms.

## Strategic Recommendations

Based on these insights, we recommend the following strategies to reduce churn:

### Strategy 1: The "Long-Term" Incentive Program
**Target:** Month-to-month customers with high monthly charges.
**Action:** Launch a targeted campaign offering a **15% discount** on the first 3 months for customers who switch to a 1-year contract.
**Goal:** Lock in high-risk customers into more stable contract terms.

### Strategy 2: "First Year" Onboarding Concierge
**Target:** New customers (Tenure < 12 months).
**Action:** Implement a proactive "Check-in" program at months 1, 3, and 6. Offer a free "Tech Support" consultation or a service optimization review.
**Goal:** Address early friction points and build loyalty during the critical first year.

### Strategy 3: Fiber Optic Value Review
**Target:** Fiber Optic subscribers.
**Action:** Conduct a survey specifically for churned Fiber users to understand if the driver is Price or Quality. Consider bundling a free streaming service (e.g., "StreamingMovies") to increase the perceived value of the higher-priced Fiber tier.
**Goal:** Reduce the ~42% churn rate in this high-value segment.

## Technical Implementation
The solution includes:
1.  **Automated Data Pipeline:** Fetches public data and generates synthetic data for testing.
2.  **Predictive Model:** A Logistic Regression classifier to score churn probability.
3.  **Interactive Dashboard:** A Streamlit app for real-time risk monitoring and reporting.
