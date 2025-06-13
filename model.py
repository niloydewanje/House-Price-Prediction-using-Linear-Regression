import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Load the California Housing dataset
california = fetch_california_housing(as_frame=True)
data = california.frame

print(data.head())  # Print first few rows

#Features and target
X = data.drop('MedHouseVal', axis=1)  # features
y = data['MedHouseVal']  # target: Median house value


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Split into train and test

#Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)  #Predict and evaluate

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plot actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted House Value")
plt.show()