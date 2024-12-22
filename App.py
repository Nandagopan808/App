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

# Generate Synthetic Dataset
def generate_data():
    np.random.seed(42)
    num_samples = 500
    data = {
        'Make': np.random.choice(['Toyota', 'Ford', 'BMW', 'Audi'], num_samples),
        'Model': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Truck'], num_samples),
        'Year': np.random.randint(2000, 2025, num_samples),
        'Mileage': np.random.uniform(5000, 150000, num_samples),  # mileage in miles
        'EngineSize': np.random.uniform(1.0, 5.0, num_samples),  # engine size in liters
        'Horsepower': np.random.uniform(100, 500, num_samples),  # horsepower
        'FuelType': np.random.choice(['Gasoline', 'Diesel', 'Electric'], num_samples),
        'Transmission': np.random.choice(['Manual', 'Automatic'], num_samples),
        'Color': np.random.choice(['Black', 'White', 'Red', 'Blue', 'Silver'], num_samples),
        'OwnerCount': np.random.randint(1, 5, num_samples),  # number of previous owners
        'Warranty': np.random.choice(['Yes', 'No'], num_samples),  # warranty availability
        'CityMPG': np.random.uniform(10, 50, num_samples),  # city mileage in MPG
        'HighwayMPG': np.random.uniform(15, 60, num_samples),  # highway mileage in MPG
        'SafetyRating': np.random.randint(1, 6, num_samples),  # safety rating out of 5
        'PopularityScore': np.random.uniform(0, 100, num_samples),  # popularity score
        'Price': np.random.randint(5000, 50000, num_samples)  # car price
    }

    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 30), 'Mileage'] = np.nan  # Missing values in Mileage
    df = pd.concat([df, df.sample(10)])  # Duplicates
    df.loc[np.random.choice(df.index, 10), 'Price'] *= 1.5  # Outliers in Price
    return df

# Preprocessing function
def preprocess_data(df):
    df_cleaned = df.drop_duplicates()
    df_cleaned['Mileage'] = df_cleaned['Mileage'].fillna(df_cleaned['Mileage'].mean())

    # Remove outliers in Price using IQR method
    q1 = df_cleaned['Price'].quantile(0.25)
    q3 = df_cleaned['Price'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_cleaned = df_cleaned[(df_cleaned['Price'] >= lower_bound) & (df_cleaned['Price'] <= upper_bound)]
    return df_cleaned

# Model training function
def train_model(df):
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ['Make', 'Model', 'FuelType', 'Transmission', 'Color', 'Warranty']
    numerical_features = ['Year', 'Mileage', 'EngineSize', 'Horsepower', 'OwnerCount', 'CityMPG', 'HighwayMPG', 'SafetyRating', 'PopularityScore']

    # Imputation and scaling
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
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred)
    r2_train = r2_score(y_train, y_pred)

    return model, mse_train, r2_train

# Streamlit Dashboard
def run_dashboard():
    st.title("Car Price Prediction Dashboard")

    # Generate and display dataset
    df = generate_data()
    st.subheader("Raw Car Dataset")
    st.dataframe(df)

    # Preprocess data and display cleaned data
    df_cleaned = preprocess_data(df)
    st.subheader("Cleaned Car Dataset (Without Duplicates, Missing Values, and Outliers)")
    st.dataframe(df_cleaned)

    # Train model
    model, mse_train, r2_train = train_model(df_cleaned)

    # Show metrics for the model
    st.subheader("Model Evaluation Metrics (Preprocessed Data)")
    st.write(f"Mean Squared Error (MSE): {mse_train:.2f}")
    st.write(f"R² Score: {r2_train:.2f}")

    # Visualizations
    st.subheader("Raw Data Distribution of Car Prices")
    fig_raw = px.histogram(df, x='Price', nbins=20, title="Raw Data Distribution")
    fig_raw.update_layout(yaxis_title="Car Price")
    st.plotly_chart(fig_raw)

    st.subheader("Cleaned Data Distribution of Car Prices")
    fig_cleaned = px.histogram(df_cleaned, x='Price', nbins=20, title="Cleaned Data Distribution")
    fig_cleaned.update_layout(yaxis_title="Car Price")
    st.plotly_chart(fig_cleaned)

    # Model Metrics Comparison
    st.subheader("Model Metrics Comparison")
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(
        name='MSE',
        x=['Raw Data', 'Preprocessed Data'],
        y=[None, mse_train]  # Raw MSE can be calculated but isn't currently done; leave as None
    ))
    fig_metrics.add_trace(go.Bar(
        name='R²',
        x=['Raw Data', 'Preprocessed Data'],
        y=[None, r2_train]  # Raw R² can be calculated similarly
    ))
    fig_metrics.update_layout(barmode='group', title="Model Performance Comparison")
    st.plotly_chart(fig_metrics)

# Run the dashboard
if __name__ == "__main__":
    run_dashboard()
