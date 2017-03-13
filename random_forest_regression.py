# RANDOM FOREST REGRESSION
# Note: Even though the random forest uses many indvidiual decision trees 
# to estimate the dependent variable, the result will NOT appraoch a continuous
# curve as you icnreas the number of trees. Instead, the law of large numbers will
# simply ensure that the average across all the trees converges towards the "true" mean
# but the step-wise behavior will remain because the final output still performs a 
# boundary-based classification. All it does is increase the statistical robustness of 
# the ordinary decision tree method







# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm
from sklearn.ensemble import RandomForestRegressor


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
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)


# Predicting a salary level of 6.5 
y_pred = regressor.predict(6.5)
print(y_pred)

# Visualize with high-res curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'r')
plt.plot(X_grid, regressor.predict(X_grid))
plt.title("Random Forest Regression")
plt.xlabel("X-axis label here")
plt.ylabel("Y-axis label here")
plt.show()















