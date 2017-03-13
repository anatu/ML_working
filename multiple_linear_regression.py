# MULTIPLE LINEAR REGRESSION

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm

# PREPROCESSING
######################################################################################

# Importing the dataset
dataset = pd.read_csv(r'50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values	

# Encode categorical data (transofrm text-categories of data into numbers)
labelencoder_X = LabelEncoder()
# Transform the country column into encoded values
# This will create a problem because it implies a priority order between categories
# since there are more than 2 categories
# (e.g. Spain=2 and Germany=1 implies Spain>Germany which is not true here)
X[:,3] = labelencoder_X.fit_transform(X[:,3])

# Encode the categorical data using a dummy variable afterwards
# categorical_features = 3 denotes the column in which the 
# LabelEncoded cateogrical features are located
onehotencoder = OneHotEncoder(categorical_features = [3])

X = onehotencoder.fit_transform(X).toarray()



# Avoid the Dummy Variable trap (only define n-1 dummy variables in the model
# because knowing the value of n-1 categories will imply the value of the nth,
# e.g. if not A or B, then it must be C
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# NAIVE MODELLINHG USING A BASIC PREDICTOR
######################################################################################
# Fitting the training dataset to the multilinear model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Use the regressor to predict the y's from the test set X's
y_pred = regressor.predict(X_test)

# OPTIMAL MODEL USING BACKWARDS ELIM.
######################################################################################
# Append a column of ones to the matrix of features so the constant b_0 is treated like a 
# parameter as b_0*x_0 where x_0 = 1 (so that it isn't ignored by the 
# backwards elimination routine we will use). Add at the start so it sits at position x_0
X = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)

# Define the optimal matrix of features which will exclude all non-significant predictors
# (Identical to X at 0th iteration)
X_opt = X[:, :]
# Declare the ordinary least-squares regressor object and fit to the data
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0,3]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())






























