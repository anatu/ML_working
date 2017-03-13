# SVR REGRESSION
# NOTE: SVR ALGORITHM DOES NOT INCLUDE FEATURE SCALING

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as sm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# PREPROCESSING
######################################################################################

# Importing the dataset
dataset = pd.read_csv(r'Position_Salaries.csv')

print(dataset.columns)

# Specify the set of predictor variables to use, and the independent variable
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values	

# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling (may throw a worning regarding conversion of data into float64 datatype)
# Now the sc_X object's transform method can be used to map values from the X domain into the 
# transformed (X') domain, and sc_y can be used to transform from y to y' (or vice versa with
# an inverse transform)
# 
# Using fit_transform fits sc_X to the observations in X, and then automatically maps from X to X'
# so that the output is already in the transformed (i.e. featured-scaled) domain 
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)



# MODELLING
######################################################################################

# Fitting the Regressor to the dataset
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


# Predicting a salary level of 6.5 (must transform the level
# to account  for feature scaling in X, then back-transform
# the result in y to get the salary estimate in real non-scaled terms)
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print(y_pred)



# Visualize the regression results 
plt.scatter(X, y, color = 'r')
plt.plot(X, regressor.predict(X))
plt.title("Title here")
plt.xlabel("X-axis label here")
plt.ylabel("Y-axis label here")
plt.show()

# Visualize with high-res curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'r')
plt.plot(X, regressor.predict(X))
plt.title("Title here")
plt.xlabel("X-axis label here")
plt.ylabel("Y-axis label here")
plt.show()















