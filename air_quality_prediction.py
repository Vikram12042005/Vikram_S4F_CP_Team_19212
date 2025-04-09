
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
# Example: air_quality.csv should include columns like PM10, NO2, CO, temperature, humidity, wind_speed, traffic_volume
data = pd.read_csv('air_quality.csv')

# Step 3: Feature selection
X = data[['PM10', 'NO2', 'CO', 'temperature', 'humidity', 'wind_speed', 'traffic_volume']]
y = data['PM2.5']  # Target: PM2.5 level (could also be AQI)

# Step 4: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 7: Output results
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("Sample Predictions:", y_pred[:5])
