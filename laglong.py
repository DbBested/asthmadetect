import pandas as pd

# Load the CSV files
addresses_with_lat_lng = pd.read_csv('C:/Users/dbbes/Downloads/merged_pm2.5FEM.csv')  # Path corrected for Windows
values_with_addresses = pd.read_csv('C:/Users/dbbes/Downloads/new_pm2.5.csv')  # Path corrected for Windows

# Capitalize the 'address' column in both dataframes for matching
addresses_with_lat_lng['ADDRESS'] = addresses_with_lat_lng['ADDRESS'].str.upper()
values_with_addresses['ADDRESS'] = values_with_addresses['ADDRESS'].str.upper()

# Merge the dataframes on the capitalized 'address' column
# Only bringing in 'latitude' and 'longitude' from the addresses_with_lat_lng dataframe
merged_df = pd.merge(values_with_addresses, addresses_with_lat_lng[['ADDRESS', 'Latitude', 'Longitude']],
                     on='ADDRESS', how='left')

# Write the merged dataframe to a new CSV file
merged_df.to_csv('C:/Users/dbbes/Downloads/merged_values_with_lat_lng.csv', index=False)
