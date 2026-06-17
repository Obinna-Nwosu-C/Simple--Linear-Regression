

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# Select features
cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]

# Extract X and y
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)

# Print coefficients
print('Coefficient (Slope):', regressor.coef_[0])
print('Intercept:', regressor.intercept_)

# Visualize training data
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.coef_[0] * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Training Data with Regression Line")
plt.show()  # ← This makes the chart appear

# Make predictions on test data
y_pred = regressor.predict(X_test.reshape(-1, 1))

# Evaluate
print("\nModel Evaluation:")
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2-score: %.2f" % r2_score(y_test, y_pred))

# Visualize test data
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, regressor.coef_[0] * X_test + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Test Data with Regression Line")
plt.show()  # ← This makes the second chart appear
