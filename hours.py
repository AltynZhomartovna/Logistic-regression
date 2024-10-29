import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

hours_studied = np.array([1, 2, 3, 4, 5, 6, 7])  # Hours studied
passed = np.array([0, 0, 0, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

hours_studied = hours_studied.reshape(-1, 1)
model = LogisticRegression()
model.fit(hours_studied, passed)

predicted_pass = model.predict([[4.5]])
print(f"Will a student who studied for 4.5 hours pass? {'Yes' if predicted_pass[0] == 1 else 'No'}")

plt.scatter(hours_studied, passed, color='blue', label='Actual data')
plt.plot(hours_studied, model.predict_proba(hours_studied)[:, 1], color='red', label='Pass Probability')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Hours Studied vs. Passing Probability')
plt.legend()
plt.show()
