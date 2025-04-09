
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Simulating sample data (replace with actual dataset)
# Columns: ['PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity', 'wind_speed', 'traffic_volume', 'AQI']

# Load dataset
# df = pd.read_csv('air_quality_data.csv')
# Simulated dummy dataset for illustration
np.random.seed(42)
df = pd.DataFrame({
    'PM2.5': np.random.uniform(30, 250, 1000),
    'PM10': np.random.uniform(50, 300, 1000),
    'NO2': np.random.uniform(10, 80, 1000),
    'CO': np.random.uniform(0.2, 3.0, 1000),
    'temperature': np.random.uniform(15, 40, 1000),
    'humidity': np.random.uniform(20, 90, 1000),
    'wind_speed': np.random.uniform(0.5, 5, 1000),
    'traffic_volume': np.random.randint(100, 1000, 1000),
    'AQI': np.random.uniform(50, 400, 1000)
})

# Feature and target split
X = df.drop(columns=['AQI'])
y = df['AQI']

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual AQI')
plt.plot(y_pred[:100], label='Predicted AQI', linestyle='--')
plt.title('AQI Prediction vs Actual (sample)')
plt.xlabel('Sample Index')
plt.ylabel('AQI')
plt.legend()
plt.grid(True)
plt.show()
