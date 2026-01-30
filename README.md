# Customer Churn Prediction & Analytics Dashboard

A machine learning application to predict customer churn, identify high-risk segments, and visualize data trends. Built with Python, Scikit-learn, and Streamlit.

## Features

*   **Interactive Dashboard:** View key metrics (Churn Rate, Revenue) and visualizations.
*   **Churn Predictor:** Input customer details to get a real-time churn probability score.
*   **High Risk Monitor:** Identify and download a list of customers with high churn risk.
*   **SQL Playground:** Execute custom SQL queries on the customer database.
*   **Strategic Report:** View retention strategies derived from data analysis.

## Setup & Installation

### Local Development

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate Data & Train Model:**
    Before running the app, you need to generate the synthetic data and train the model.
    ```bash
    python src/data_generator.py
    python src/model_train.py
    ```

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

## Deployment on Render

This application is ready to be deployed on [Render](https://render.com).

1.  **Create a New Web Service** on Render.
2.  **Connect your repository.**
3.  **Configure the build and start commands:**

    *   **Build Command:** `pip install -r requirements.txt && python src/data_generator.py && python src/model_train.py`
        *(Note: The data generation and training happen during build to ensure the model is ready for the app)*
    *   **Start Command:** `streamlit run app.py`

4.  **Deploy!**

## Project Structure

*   `app.py`: Main Streamlit application.
*   `src/`: Source code for data generation, analysis, and model training.
*   `data/`: Generated synthetic data (CSV).
*   `models/`: Saved model artifacts (.pkl files).
*   `notebooks/`: Jupyter notebooks and plot images.
*   `REPORT.md`: Comprehensive analysis and strategy report.
