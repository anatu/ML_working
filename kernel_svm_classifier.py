# SUPPORT VECTOR MACHINE CLASSIFICATION, WITH VARYING KERNELS
# NOTE: SVM classifiers require feature scaling! Remember to back-transform
# using the scaler object in order to interpret results in their original units

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# PREPRPOCESSING
########################################################################
# Importing the dataset
dataset = pd.read_csv(r'Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)


# MODELLING
########################################################################

# Fitting the classifier of choice to the Training set
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Create a Confusion Matrix to assess the performance of the predictions
# i.e. to determine the number of accurate vs. inaccurate predictions
# outputted by the logistic classifier

# Mathematical def'n, C s.t. C(i,j) is the number of points known to 
# be in group i and predicted to be in group j, across all categories
# of the dependent variable y. For a Yes/No categorical we have a 2x2 output

# This means the number of CORRECT predictions occur at cells when i=j, 
# so ALONG THE MAIN DIAGONAL OF THE MATRIX!!!!
# Therefore number of correct prediction
cm = confusion_matrix(y_test, y_pred)
print(cm)




# VISUALIZATION
########################################################################

# Visualizing the Training set results
####################################

from matplotlib.colors import ListedColormap

# Create local variables X_set and y_set for use in plotting
X_set, y_set = X_train, y_train

# Create a mesh of values of each feature X1 and X2, i.e. each column of the 
# matrix of features X
# Two np.arange calls are employed creating a range of step 0.01 for each feature
# spanning from 1 less than the minimum real observation to 1 more than the maximum observation
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), \
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# Use the classifier to predict the outcome of each point in the mesh
# The ravel() call simply flattens the meshes created above into individual lists,
# then they are packaged back into a numpy array and fed into the predictor
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Scatter the real observations onto the plot using the same color scheme
# used to generate the background mesh colors 
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support Vector Machine Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




# Visualising the Test set results
####################################

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support Vector Machine Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Visualizing the test set results with back-transformed units for features
####################################
# NOTES: Classifier is fitted to scaled features, so need to re-transform
# any features into scaled units that are fed into classifier.predict()
# (i.e. when plotting the prediction regions as a mesh)

# Find a way to scale the step size of hte background mesh color points
# based on the un-scaled feature units (scaled mesh range is 
# min - 1 to max + 1, step = 0.01 but that step size might be too granular
# and cause a crash depending on the units of the feature and range of values)

from matplotlib.colors import ListedColormap

# Back-transform the original working sets into their un-scaled units
X_set, y_set = sc_X.inverse_transform(X_test).astype(int), y_test

# Create a mesh of features in their un-scaled units
# Ranging from the min value - 1 to max value + 1
# TODO: Define step size relative to feature units!
# X1 and X2 are each mesh arrays of the two different features in X_set
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 1))


# Plot the prediction regions, re-transfomring the features in order
# to allow the predictor to make the correct predictions and thus generate
# the correct prediction regions on un-scaled feature axes

contour_points = np.array([X1.ravel(), X2.ravel()]).T
contour_points = sc_X.transform(contour_points)
print(X1.shape)

plt.contourf(X1, X2, classifier.predict(contour_points).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support Vector Machine Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

