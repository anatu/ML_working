# LOGISTIC REGRESSION

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



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

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
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
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results

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
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
