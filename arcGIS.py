import pandas as pd
from joblib import load
from arcgis.gis import GIS
from arcgis.features import FeatureLayer

# Your credentials (use with caution)
username = 'dbbested@gmail.com'
password = 'iamnightblue3'  # It's better to use getpass.getpass() to prompt for a password

# Authenticate with your ArcGIS account
gis = GIS('https://www.arcgis.com', username, password)

