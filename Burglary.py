from sklearn.naive_bayes import GaussianNB
import numpy as np

# Dataset: [Burglary, Earthquake] -> Alarm
# 1 = True, 0 = False
data = np.array([
    [1, 0, 1],  # Burglary, No Earthquake -> Alarm
    [0, 1, 1],  # No Burglary, Earthquake -> Alarm
    [1, 1, 1],  # Burglary, Earthquake -> Alarm
    [0, 0, 0],  # No Burglary, No Earthquake -> No Alarm
    [1, 0, 1],  # Burglary, No Earthquake -> Alarm
    [0, 0, 0]   # No Burglary, No Earthquake -> No Alarm
])

# Features (Burglary, Earthquake) and Labels (Alarm)
X = data[:, :2]  # Features
y = data[:, 2]   # Labels

# Train Naive Bayes Classifier
model = GaussianNB()
model.fit(X, y)

# Test the model
test_data = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])  # Test cases
predictions = model.predict(test_data)

# Output predictions
for i, test in enumerate(test_data):
    print(f"Input: Burglary={test[0]}, Earthquake={test[1]} -> Alarm Prediction: {predictions[i]}")

