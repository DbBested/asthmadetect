import pandas as pd


addresses_with_lat_lng = pd.read_csv('C:/Users/dbbes/Downloads/merged_pm2.5FEM.csv') 
values_with_addresses = pd.read_csv('C:/Users/dbbes/Downloads/new_pm2.5.csv') 


addresses_with_lat_lng['ADDRESS'] = addresses_with_lat_lng['ADDRESS'].str.upper()
values_with_addresses['ADDRESS'] = values_with_addresses['ADDRESS'].str.upper()


merged_df = pd.merge(values_with_addresses, addresses_with_lat_lng[['ADDRESS', 'Latitude', 'Longitude']],
                     on='ADDRESS', how='left')

merged_df.to_csv('C:/Users/dbbes/Downloads/merged_values_with_lat_lng.csv', index=False)
