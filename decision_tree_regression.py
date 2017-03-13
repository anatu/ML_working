# DECISION TREE REGRESSION
# NOTE: The decision tree will produce outputs according to the average value between
# the sub-intervals that it chooses, i.e. the value outputted by a predictor between
# any two real observations CANNOT be continuously increasing/decreasing,
# since it is always computing the prediction by taking the mean of real data points.
# The predictor plot should always resemble a step function

# So, if you produce a decision tree regression that has a non-horizontal line 
# running between two points, this is an indication that you are not producing prediction
# points on a fine enough granularity so the plot will simply draw straight lines between
# the points that it renders, making it seem as though the decision tree is predicting 
# a non-zero slope between two points whne in reality it is only plotting at the start
# and end points, and the plt.plot() function simply draws straight lines between the rendered points


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm
from sklearn.tree import DecisionTreeRegressor



# PREPROCESSING
######################################################################################

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\anatu\Desktop\ZZ_Tech\ML_working\Part 2\Position_Salaries.csv')

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
# so that the output is already in the transformed (i.e. featured-scaled) domain	q 
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X = sc_X.fit_transform(X)
# y = sc_y.fit_transform(y)



# MODELLING
######################################################################################

# Fitting the Regressor to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# Predicting a salary level of 6.5 
y_pred = regressor.predict(6.5)
print(y_pred)

# Visualize with high-res curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'r')
plt.plot(X_grid, regressor.predict(X_grid))
plt.title("Decision Tree Regression")
plt.xlabel("X-axis label here")
plt.ylabel("Y-axis label here")
plt.show()















