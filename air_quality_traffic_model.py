
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset (Replace with actual path or use real-time data pipeline)
data = pd.read_csv("urban_air_quality.csv")  # Example columns: PM2.5, PM10, NO2, CO, Temp, Humidity, TrafficVolume, AQI

# Feature selection
features = data[['PM2.5', 'PM10', 'NO2', 'CO', 'Temp', 'Humidity', 'TrafficVolume']]
target = data['AQI']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predict AQI on test set
predicted = model.predict(X_test)
print("Mean Squared Error on Test Data:", mean_squared_error(y_test, predicted))

# Traffic decision logic based on predicted AQI
def manage_traffic(aqi):
    if aqi >= 200:
        return "High AQI - Restrict vehicles in critical zones, enable carpool-only lanes"
    elif aqi >= 150:
        return "Moderate AQI - Adjust traffic lights to reduce congestion, promote public transport"
    else:
        return "AQI Normal - Maintain standard traffic flow"

# Simulate prediction for a new observation
sample_input = np.array([[190, 220, 80, 1.5, 30, 50, 1300]])  # Replace with real-time sensor values
predicted_aqi = model.predict(sample_input)[0]
print("Predicted AQI for current conditions:", predicted_aqi)
print("Traffic Management Suggestion:", manage_traffic(predicted_aqi))
