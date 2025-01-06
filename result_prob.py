import pandas as pd
from joblib import load


model_path = r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\random_forest_model.joblib"
rf_model = load(model_path)


features = ['NO2_KRIG', 'O2_KRIG', 'pm25v2_KRIG', 'pm25v1_KRIG', 'NEAREST_OPENSPACE', 'NEAREST_ROAD']


asthma_file_path = r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\asthma_points_with_pos.csv"
non_asthma_file_path = r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\non_asthma_points_with_pos.csv"

asthma_df = pd.read_csv(asthma_file_path)
non_asthma_df = pd.read_csv(non_asthma_file_path)


non_asthma_df.rename(columns={'Longitutde_First': 'Longitude', 'Latitude_Last': 'Latitude', 'NEAR_OPENSPACE': 'NEAREST_OPENSPACE', 'NEAR_ROAD': 'NEAREST_ROAD', 'Prevalence': 'prevalence_int'}, inplace=True)
asthma_df.rename(columns={'Longitude_Last': 'Longitude', 'Latitude_Last': 'Latitude', 'NEAREST_OPENSPACE': 'NEAREST_OPENSPACE', 'NEAREST_ROAD': 'NEAREST_ROAD', 'prevalence_int': 'prevalence_int'}, inplace=True)


combined_df = pd.concat([asthma_df, non_asthma_df], ignore_index=True)


combined_df.dropna(subset=features + ['Latitude', 'Longitude'], inplace=True)


X = combined_df[features]


combined_df['asthma_hotspot_probability'] = rf_model.predict_proba(X)[:, 1]


output_columns = features + ['Latitude', 'Longitude', 'asthma_hotspot_probability']
output_df = combined_df[output_columns]


output_path = r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\asthma_probabilities_with_pos.csv"
output_df.to_csv(output_path, index=False)
print(f"Probabilities with coordinates and original features saved to {output_path}")
