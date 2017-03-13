# SIMPLE LINEAR REGRESSION

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv(r'Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

# Fitting simple linear regression to the training set (calculates B1 and B2 coeffs
# using least-squares regression)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the results of the Test set
# By using the FITTED regressor object to estimate y values for X_test
# to compare to y_test
y_pred = regressor.predict(X_test)

# VISUALIZE TRAINING SET RESULTS
# Scatter-plot the raw training data
plt.scatter(X_train, y_train, color='red')
# Line-plot the trendline fitted to training data
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs. Experience (Training Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# VISUALIZE THE TEST SET RESULTS
# Scatter-plot the test set observations
plt.scatter(X_test, y_test, color='red')
# Line-plot the trendline fitted to TRAINING data
# To see how well it correlates to the test set observations
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs. Experience (Test Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
