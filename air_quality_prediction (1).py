
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('urban_air_quality.csv')  # Simulated dataset with PM2.5, PM10, NO2, CO, traffic, temp, humidity, wind

# Features and target
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'traffic_volume', 'temperature', 'humidity', 'wind_speed']
target = 'AQI'

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Predicted AQI values:", y_pred[:5])
print("Mean Squared Error:", mse)
