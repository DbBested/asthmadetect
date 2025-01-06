import pandas as pd
from joblib import load
from arcgis.gis import GIS
from arcgis.features import FeatureLayer


username = ''
password = ''

gis = GIS('https://www.arcgis.com', username, password)

