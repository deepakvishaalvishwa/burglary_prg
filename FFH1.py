import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Sample data: two groups of points (features in 2D)
X = np.array([
    [1, 2], [2, 3], [3, 3], [6, 1], [7, 2], # Group 0 (blue)
    [6, 5], [7, 7], [8, 6], [7, 5], [6, 6] # Group 1 (red)
])


# Labels: 0 or 1 corresponding to the two groups
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Create and train linear SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Create a grid to plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict class labels for every point in the grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary by coloring regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Plot the original data points with color coding
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Group 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Group 1')

# Highlight the support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM: Separating Two Groups with a Decision Boundary')
plt.legend()
plt.show()
