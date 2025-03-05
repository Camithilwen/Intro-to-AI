import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset (assuming you have a CSV file)
data = pd.read_csv("house_prices.csv")

# Prepare the data (selecting features and target variable)
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
y = data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print the model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict the price of a new house
new_house = np.array([[2000, 3, 2, 0]])
predicted_price = model.predict(new_house)
print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")
