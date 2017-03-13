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

# Specify the set of predictor variables to use, and the independent variable
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values	

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""



# MODELLING
######################################################################################

# Fitting the Regressor to the dataset
# MAKE YOUR REGRESSOR HERE

# Predicting an individual result with the regressor (interpolate or extrapolate)
y_pred = regressor.predict(6.5)

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















