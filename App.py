
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# Step 1: Streamlit URL Input
st.title("Car Price Prediction Model")

# Input URLs for Raw Data and Preprocessed Data
raw_data_url = st.text_input("https://raw.githubusercontent.com/Nandagopan808/App/refs/heads/main/raw_data.csv")
preprocessed_data_url = st.text_input("Enter the URL for Preprocessed Data (CSV):")

# Load Raw Data
if raw_data_url:
    try:
        df_raw = pd.read_csv(raw_data_url)
        st.write("Raw Data Preview:")
        st.write(df_raw.head())

        # Proceed with preprocessing and model training
        X = df_raw.drop(columns=['Price'])
        y = df_raw['Price']
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing
        categorical_features = ['Make', 'Model', 'FuelType', 'Transmission', 'Color', 'Warranty']
        numerical_features = ['Year', 'Mileage', 'EngineSize', 'Horsepower', 'OwnerCount', 'CityMPG', 'HighwayMPG', 'SafetyRating', 'PopularityScore']

        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        # Train model on raw data
        model.fit(X_train_raw, y_train_raw)
        y_pred_raw = model.predict(X_train_raw)

        # Evaluate raw model
        mse_raw = mean_squared_error(y_train_raw, y_pred_raw)
        r2_raw = r2_score(y_train_raw, y_pred_raw)

        st.write(f"Raw Data - Mean Squared Error (MSE): {mse_raw:.2f}")
        st.write(f"Raw Data - R² Score: {r2_raw:.2f}")

        # Display raw data distribution plot
        fig_raw = px.histogram(df_raw, x='Price', nbins=20, title="Raw Data Distribution")
        fig_raw.update_layout(yaxis_title="Car Price")
        st.plotly_chart(fig_raw)

    except Exception as e:
        st.error(f"Error loading raw data from URL: {e}")

# Load Preprocessed Data
if preprocessed_data_url:
    try:
        df_cleaned = pd.read_csv(preprocessed_data_url)
        st.write("Preprocessed Data Preview:")
        st.write(df_cleaned.head())

        # Remove duplicates and handle missing values
        df_cleaned = df_cleaned.drop_duplicates()
        df_cleaned['Mileage'] = df_cleaned['Mileage'].fillna(df_cleaned['Mileage'].mean())

        # Remove outliers in Price using IQR method
        q1 = df_cleaned['Price'].quantile(0.25)
        q3 = df_cleaned['Price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df_cleaned = df_cleaned[(df_cleaned['Price'] >= lower_bound) & (df_cleaned['Price'] <= upper_bound)]

        X_cleaned = df_cleaned.drop(columns=['Price'])
        y_cleaned = df_cleaned['Price']

        X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)

        mse_train = mean_squared_error(y_train, y_pred)
        r2_train = r2_score(y_train, y_pred)

        st.write(f"Preprocessed Data - Mean Squared Error (MSE): {mse_train:.2f}")
        st.write(f"Preprocessed Data - R² Score: {r2_train:.2f}")

        # Display cleaned data distribution plot
        fig_cleaned = px.histogram(df_cleaned, x='Price', nbins=20, title="Cleaned Data Distribution")
        fig_cleaned.update_layout(yaxis_title="Car Price")
        st.plotly_chart(fig_cleaned)

        # Model Performance Comparison
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(
            name='MSE',
            x=['Raw Data', 'Preprocessed Data'],
            y=[mse_raw, mse_train]
        ))
        fig_metrics.add_trace(go.Bar(
            name='R²',
            x=['Raw Data', 'Preprocessed Data'],
            y=[r2_raw, r2_train]
        ))
        fig_metrics.update_layout(barmode='group', title="Model Performance Comparison")
        st.plotly_chart(fig_metrics)

    except Exception as e:
        st.error(f"Error loading preprocessed data from URL: {e}")
