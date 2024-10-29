import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X = np.array([
    [0.1, 100], [0.3, 200], [0.2, 150], [0.8, 250],
    [0.9, 300], [0.7, 250], [0.6, 200], [0.4, 100],
    [0.3, 120], [0.7, 210]
])

y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])

model = LogisticRegression()
model.fit(X, y)

predictions = model.predict(X)
train_accuracy = accuracy_score(y, predictions)

plt.figure(figsize=(10, 6))

x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label="Not Spam", edgecolor='k')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label="Spam", edgecolor='k')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Logistic Regression Decision Boundary (Accuracy: {train_accuracy*100:.2f}%)")
plt.legend()
plt.show()
