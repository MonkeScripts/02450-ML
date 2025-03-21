# exercise 6.3.2

import numpy as np

# requires data from exercise 1.5.1
from ex1_5_1 import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

# Maximum number of neighbors
L = 40

CV = model_selection.LeaveOneOut()
errors = np.zeros((N, L))
i = 0
for train_index, test_index in CV.split(X, y):
    print("Crossvalidation fold: {0}/{1}".format(i + 1, N))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1, L + 1):
        knclassifier = KNeighborsClassifier(n_neighbors=l)
        knclassifier.fit(X_train, y_train)
        y_est = knclassifier.predict(X_test)
        errors[i, l - 1] = np.sum(y_est[0] != y_test[0])

    i += 1

# Plot the classification error rate
plt.figure()
plt.plot(100 * sum(errors, 0) / N)
plt.xlabel("Number of neighbors")
plt.ylabel("Classification error rate (%)")
plt.show()

print("Ran Exercise 6.3.2")
