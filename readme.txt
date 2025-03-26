# Solar Power Generation Prediction App

## Overview
This is a **Streamlit** web application that predicts solar power generation based on user-provided environmental parameters. The model uses a **Random Forest Regressor** trained on sample data.

## Features
- **User Input:**
  - Temperature (°C)
  - Humidity (%)
  - Solar Radiation (W/m²)
  - Wind Speed (m/s)
- **Prediction:**
  - The app predicts the solar power output in **kW** based on user inputs.
- **Model Evaluation:**
  - Users can view model performance metrics including:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - R-squared (R2) score

## Technologies Used
- **Python**
- **Streamlit** (for web interface)
- **Pandas** (for data manipulation)
- **NumPy** (for numerical operations)
- **Scikit-learn** (for machine learning model)

## Installation
1. Clone the repository or copy the script.
2. Install the required dependencies:
   ```sh
   pip install streamlit pandas numpy scikit-learn
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Code Explanation
1. **Data Preparation:**
   - A sample dataset is created with temperature, humidity, solar radiation, wind speed, and power output.
   - The dataset is split into training and testing sets.
2. **Model Training:**
   - A `RandomForestRegressor` model is trained on the dataset.
3. **Streamlit Interface:**
   - Users can input environmental parameters using sliders.
   - The model predicts solar power output based on inputs.
   - Model evaluation metrics are displayed if selected.

## Future Enhancements
- Use a real-world dataset for better accuracy.
- Allow users to upload their own dataset.
- Deploy the app on a cloud platform like **Heroku** or **AWS**.

## License
This project is open-source and available under the **MIT License**.


