import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Streamlit Title
st.title("House Price Prediction with Raw and Preprocessed Data")

# Load raw and preprocessed datasets from URLs
raw_data_path = "https://raw.githubusercontent.com/Nandagopan808/App/refs/heads/main/preprocessed_data.csv"
preprocessed_data_path = "https://raw.githubusercontent.com/Nandagopan808/App/refs/heads/main/preprocessed_data.csv"

@st.cache_data
def load_data(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

# Load data
raw_data = load_data(raw_data_path)
preprocessed_data = load_data(preprocessed_data_path)

# Display raw and preprocessed datasets
st.write("## Raw Dataset")
if raw_data is not None:
    st.dataframe(raw_data.head())
else:
    st.warning("Raw data not available!")

st.write("## Preprocessed Dataset")
if preprocessed_data is not None:
    st.dataframe(preprocessed_data.head())
else:
    st.warning("Preprocessed data not available!")

# Check if both datasets are loaded for further processing
if raw_data is not None and preprocessed_data is not None:
    # Step 2: Prepare data for modeling
    # Set target variable
    target_column = "Price"
    if target_column not in raw_data.columns or target_column not in preprocessed_data.columns:
        st.error(f"Target column '{target_column}' not found in datasets!")
    else:
        X_raw = raw_data.drop(columns=[target_column])
        y_raw = raw_data[target_column]

        X_preprocessed = preprocessed_data.drop(columns=[target_column])
        y_preprocessed = preprocessed_data[target_column]

        # Step 3: Split data into training and test sets
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=42
        )
        X_train_prep, X_test_prep, y_train_prep, y_test_prep = train_test_split(
            X_preprocessed, y_preprocessed, test_size=0.2, random_state=42
        )

        # Step 4: Define preprocessing and training pipelines
        categorical_features = [col for col in X_raw.columns if X_raw[col].dtype == "object"]
        numerical_features = [col for col in X_raw.columns if X_raw[col].dtype in ["int64", "float64"]]

        numerical_transformer = Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", RobustScaler()),
        ])
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42)),
        ])

        # Step 5: Train models and make predictions
        model.fit(X_train_raw, y_train_raw)
        y_pred_raw = model.predict(X_test_raw)
        mse_raw = mean_squared_error(y_test_raw, y_pred_raw)
        r2_raw = r2_score(y_test_raw, y_pred_raw)

        model.fit(X_train_prep, y_train_prep)
        y_pred_prep = model.predict(X_test_prep)
        mse_prep = mean_squared_error(y_test_prep, y_pred_prep)
        r2_prep = r2_score(y_test_prep, y_pred_prep)

        # Display model performance
        st.write("## Model Performance")
        st.write(f"### Raw Data: Mean Squared Error = {mse_raw:.2f}, R² Score = {r2_raw:.2f}")
        st.write(f"### Preprocessed Data: Mean Squared Error = {mse_prep:.2f}, R² Score = {r2_prep:.2f}")

        # Step 6: Visualizations
        st.write("## Visualizations")
        # Performance Comparison Bar Plot
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(
            name="MSE",
            x=["Raw Data", "Preprocessed Data"],
            y=[mse_raw, mse_prep]
        ))
        fig_metrics.add_trace(go.Bar(
            name="R² Score",
            x=["Raw Data", "Preprocessed Data"],
            y=[r2_raw, r2_prep]
        ))
        fig_metrics.update_layout(
            barmode="group",
            title="Model Performance Comparison",
            xaxis_title="Dataset",
            yaxis_title="Performance Metrics",
        )
        st.plotly_chart(fig_metrics)

        # Raw vs Preprocessed Price Distribution
        fig_raw = px.histogram(raw_data, x=target_column, nbins=20, title="Raw Data Price Distribution")
        st.plotly_chart(fig_raw)

        fig_prep = px.histogram(preprocessed_data, x=target_column, nbins=20, title="Preprocessed Data Price Distribution")
        st.plotly_chart(fig_prep)
else:
    st.warning("Ensure both raw and preprocessed datasets are available to proceed!")
