import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
asthma_df = pd.read_csv(r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\asthma_points_simplified.csv")
non_asthma_df = pd.read_csv(r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\non_asthma_points.csv")

# Prepare the data
asthma_df['target'] = 1
non_asthma_df['target'] = 0
combined_df = pd.concat([asthma_df, non_asthma_df])

features = ['NO2_KRIG', 'O2_KRIG', 'pm25v2_KRIG', 'pm25v1_KRIG', 'NEAREST_OPENSPACE', 'NEAREST_ROAD']
combined_df.dropna(subset=features, inplace=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

X = combined_df[features]
y = combined_df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_models(X_train, X_test, y_train, y_test, param_values, param_name):
    train_accuracies = []
    test_accuracies = []
    for value in param_values:
        kwargs = {param_name: value, 'random_state': 42}
        model = RandomForestClassifier(**kwargs)
        model.fit(X_train, y_train)
        train_accuracies.append(model.score(X_train, y_train))
        test_accuracies.append(model.score(X_test, y_test))
    return train_accuracies, test_accuracies

# n_estimators exploration remains the same
n_estimators_values = range(10, 151, 5)

# Adjust max_depth exploration to end at 15
max_depth_values = range(1, 16)  # Adjusted to end at 15

# Evaluate n_estimators impact
n_estimators_train, n_estimators_test = evaluate_models(X_train, X_test, y_train, y_test, n_estimators_values, 'n_estimators')
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(n_estimators_values, n_estimators_train, label='Training Accuracy', marker='o')
plt.plot(n_estimators_values, n_estimators_test, label='Testing Accuracy', marker='x')
plt.title('Accuracy vs. n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend()

# Evaluate max_depth impact with the adjusted range
max_depth_train, max_depth_test = evaluate_models(X_train, X_test, y_train, y_test, max_depth_values, 'max_depth')
plt.subplot(1, 2, 2)
plt.plot(max_depth_values, max_depth_train, label='Training Accuracy', marker='o')
plt.plot(max_depth_values, max_depth_test, label='Testing Accuracy', marker='x')
plt.title('Accuracy vs. max_depth')
plt.xlabel('max_depth')
plt.legend()

plt.tight_layout()
plt.show()
