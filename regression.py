import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump


data_path = r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\asthma_points_with_pos.csv"
df = pd.read_csv(data_path)


features = ['NO2_KRIG', 'O2_KRIG', 'pm25v2_KRIG', 'pm25v1_KRIG', 'NEAREST_OPENSPACE', 'NEAREST_ROAD']
target = 'prevalence_int'


df = df.dropna(subset=features + [target])


X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)


rf_regressor.fit(X_train, y_train)


y_train_pred = rf_regressor.predict(X_train)
y_test_pred = rf_regressor.predict(X_test)


train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse:.4f}, Training R^2: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, Test R^2: {test_r2:.4f}")

model_path = r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\random_forest_regressor.joblib"
dump(rf_regressor, model_path)
print(f"Model saved to {model_path}")


feature_importances = rf_regressor.feature_importances_
sorted_importances = sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True)
print("Feature Importances:")
for feature, importance in sorted_importances:
    print(f"{feature}: {importance:.4f}")
