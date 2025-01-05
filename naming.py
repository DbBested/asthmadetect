from fuzzywuzzy import process, fuzz
import pandas as pd
import geopandas as gpd

# Load your CSV and shapefile
csv_data = pd.read_csv("C:\\Users\\dbbes\\Downloads\\pediatric (1).csv")
shape_data = gpd.read_file("C:\\Users\\dbbes\\Downloads\\schools\\Schools\\SCHOOLS_PT.shp")

# Function to apply fuzzy matching
def get_best_match(name, choices, scorer, limit=1):
    return process.extractOne(name, choices, scorer=scorer)

# Applying the fuzzy matching to each school name in the CSV
choices = shape_data['NAME'].tolist()  # Replace 'Name_Field' with the actual field name of the school in your shapefile
csv_data['matched_name'] = csv_data['School'].apply(lambda x: get_best_match(x, choices, fuzz.ratio)[0])


# Optionally, you can merge the dataframes based on the matched names
# merged_data = pd.merge(csv_data, shape_data, left_on='matched_name', right_on='Name_Field')

# Save the updated CSV data to a new file
csv_data.to_csv('updated_csv.csv', index=False)

# Print a message indicating completion
print("Script has completed and updated_csv.csv has been created.")
