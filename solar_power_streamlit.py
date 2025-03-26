import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Sample data generation (replace with real data)
data = {
    'temperature': [25, 30, 22, 35, 28, 24, 33, 29, 26, 31],
    'humidity': [60, 65, 55, 70, 50, 60, 55, 63, 67, 61],
    'solar_radiation': [400, 500, 300, 600, 450, 350, 550, 500, 450, 520],
    'wind_speed': [5, 10, 4, 8, 6, 7, 9, 8, 6, 5],
    'power_output': [5, 7, 3, 9, 6, 4, 8, 7, 6, 7]  # Power output in kW
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Split dataset into features and target
X = df[['temperature', 'humidity', 'solar_radiation', 'wind_speed']]
y = df['power_output']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("Solar Power Generation Prediction")

# User Inputs
temperature = st.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
solar_radiation = st.slider("Solar Radiation (W/m²)", min_value=0, max_value=1000, value=500)
wind_speed = st.slider("Wind Speed (m/s)", min_value=0, max_value=20, value=5)

# Create feature array from user input
user_input = np.array([[temperature, humidity, solar_radiation, wind_speed]])

# Predict the solar power output
prediction = model.predict(user_input)

# Display the prediction
st.write(f"Predicted Solar Power Output: {prediction[0]:.2f} kW")

# Show model metrics
if st.checkbox("Show Model Evaluation Metrics"):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R2): {r2:.2f}")

