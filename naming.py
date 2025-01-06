from fuzzywuzzy import process, fuzz
import pandas as pd
import geopandas as gpd


csv_data = pd.read_csv("C:\\Users\\dbbes\\Downloads\\pediatric (1).csv")
shape_data = gpd.read_file("C:\\Users\\dbbes\\Downloads\\schools\\Schools\\SCHOOLS_PT.shp")


def get_best_match(name, choices, scorer, limit=1):
    return process.extractOne(name, choices, scorer=scorer)


choices = shape_data['NAME'].tolist() 
csv_data['matched_name'] = csv_data['School'].apply(lambda x: get_best_match(x, choices, fuzz.ratio)[0])





csv_data.to_csv('updated_csv.csv', index=False)


print("Script has completed and updated_csv.csv has been created.")
