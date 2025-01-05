import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from joblib import dump
# Load the datasets
asthma_df = pd.read_csv(r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\asthma_points_simplified.csv")
non_asthma_df = pd.read_csv(r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\non_asthma_points.csv")

# Assuming 'Prevalence' column exists, and prevalence > 0 indicates asthma points
asthma_df['target'] = 1
non_asthma_df['target'] = 0
non_asthma_df['Prevalence'] = 0  # Assign 0 to 'Prevalence' for non-asthma points

# Combine the datasets
combined_df = pd.concat([asthma_df, non_asthma_df])

# Clean 'NEAREST' columns, remove the leading ' and convert to float


# Select features and target
features = ['NO2_KRIG', 'O2_KRIG', 'pm25v2_KRIG', 'pm25v1_KRIG', 'NEAREST_OPENSPACE', 'NEAREST_ROAD']
combined_df.dropna(subset=features, inplace=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index()


X = combined_df[features]
y = combined_df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_clf.fit(X_train, y_train)
y_train_pred = rf_clf.predict(X_train)
model_path = r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\random_forest_model.joblib"
dump(rf_clf, model_path)
print(f"Model saved to {model_path}")

# Calculate training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Predict on the test set
y_pred = rf_clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print metrics
print(f'Accuracy: {accuracy:.5f}')
print(f'Precision: {precision:.5f}')
print(f'Recall: {recall:.5f}')
print(f'F1 Score: {f1:.5f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
importances = rf_clf.feature_importances_

# Convert the importances into a more readable format and sort them
feature_importance_dict = {feature: importance for feature, importance in zip(features, importances)}
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

# Print the feature importances
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance:.4f}")

probabilities = rf_clf.predict_proba(combined_df[features])[:, 1]

# Add the probabilities to the dataframe
combined_df['asthma_hotspot_probability'] = probabilities

# Save the dataframe with probabilities to a new CSV file
output_path = r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\asthma_probabilities.csv"
combined_df.to_csv(output_path, index=False)
print(f"Probabilities saved to {output_path}")
importances = rf_clf.feature_importances_

# Convert the importances into a readable format
features_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

# Print the significance of each feature
print("Significance of each feature/column:")
for feature, importance in features_importance:
    print(f"{feature}: {importance:.4f}")