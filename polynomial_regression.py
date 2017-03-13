# POLYNOMIAL REGRESSION

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

# PREPROCESSING
######################################################################################

# Importing the dataset
dataset = pd.read_csv(r'Position_Salaries.csv')

print(dataset.columns)

# Do not use the Position column in X since it's not a predictor - only the Level is
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values	

# NO ENCODING - NO CATEGORICAL VARIABLES
# # Encode categorical data (transofrm text-categories of data into numbers)
# labelencoder_X = LabelEncoder()
# # Transform the country column into encoded values
# # This will create a problem because it implies a priority order between categories
# # since there are more than 2 categories
# # (e.g. Spain=2 and Germany=1 implies Spain>Germany which is not true here)
# X[:,3] = labelencoder_X.fit_transform(X[:,3])
# # Encode the categorical data using a dummy variable afterwards
# # categorical_features = 3 denotes the column in which the 
# # LabelEncoded cateogrical features are located
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()
# # Avoid the Dummy Variable trap (only define n-1 dummy variables in the model
# # because knowing the value of n-1 categories will imply the value of the nth,
# # e.g. if there are A,B,C and it's not A or B, then it must be C
# X = X[:, 1:]

# NO TRAIN/TEST SPLIT - N IS TOO SMALL - 
# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# MODELLING
######################################################################################

# Create a simple linear regressor 
lin_reg = LinearRegression()
lin_reg.fit(X,y)
 
# Create a polynomial linear regressor of degree 2 (quadratic)
# Fit poly_reg to X and transform X into X_poly, which adds the quadratic feature to the dataset
# So that X contains column[s] for the relevant degrees (e.g. X^2, X^3 ....)  
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Higher-degree polynomial fit
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Plot the simple linear trendline with the observations
plt.scatter(X, y, color = 'g')
plt.plot(X, lin_reg.predict(X), color = 'r')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Plot the polynomial trend curve with the observations
plt.scatter(X, y, color = 'g')
# (Need to fit_transform the X data in the predict function so that it fits
# the data using the polynomial curve rather than a straight line)
# Increase the resolution of the trend curve by using arange, 
# then use reshape to convert from vector into matrix datatype
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'b')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predict a result with simple linear regression (interpolation)
print(lin_reg.predict(6.5))

# Predict a new result (can use transform or fit_transform - same result 
print(lin_reg_2.predict(poly_reg.transform(6.5)))



















