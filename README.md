This project aims to build a machine learning model to predict car prices based on various features such as make, model, year, engine size, mileage, and more. 
The goal is to provide accurate predictions that could help individuals or businesses assess the value of a used car.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
file_path = 'C:/Users/PMLS/Downloads/car data.csv'
df = pd.read_csv(file_path)
df.info()
df.describe()
df.isnull().sum
df_clean = df.dropna()
# Convert categorical columns into dummy/indicator variables
df_clean = pd.get_dummies(df_clean, drop_first=True)
X = df_clean.drop('Selling_Price', axis=1)  # Assuming 'price' is the target column
y = df_clean['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Mean Squared Error: {mse_rf}')
print(f'Random Forest R-squared: {r2_rf}')


import joblib

# Save the trained model
joblib.dump(rf_model, 'car_price_prediction_model.pkl')
