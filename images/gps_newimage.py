# %%
# Python program to get a google map 
# image of specified location using 
# Google Static Maps API
  
# importing required modules
import requests
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import pandas as pd

from PIL import Image

## save place

realsat_shape_file_path ="/home/kumarv/pravirat/Realsat_labelling/river_code_kx/351_400_shape_file/ReaLSAT_351_400_only.shp"
# Enter your api key here
api_key = "AIzaSyDA1UA8IRqJXQp0Rk10kwZL0GpJXqTl3kU"
  
# url variable store url
url = "https://maps.googleapis.com/maps/api/staticmap?"
  
# center defines the center of the map,
# equidistant from all edges of the map. 
center = "= "
  
# zoom defines the zoom
# level of the map
zoom = 10
size = '640x640'
#location = '21.644705,109.748304'


## ID list
ID_list = []

## load the .shp file
gdf1 =  gpd.read_file(realsat_shape_file_path)

for id in ID_list:
    for i in range(0, gdf1.shape[0]):
        if id == gdf1['ID'].iloc[i]:
            x_coordinate = round(gdf1['geometry'].iloc[i].centroid.x,7)
            y_coordinate = round(gdf1['geometry'].iloc[i].centroid.y,7)
            location = str(y_coordinate) + ',' + str(x_coordinate)
 
            response = requests.get('https://maps.googleapis.com/maps/api/staticmap?center='+str(location)+'&zoom=16&size=640x640&maptype=satellite&key=AIzaSyDA1UA8IRqJXQp0Rk10kwZL0GpJXqTl3kU')
    
            if response.status_code== 200:
                name = 'GPS_ID_'+ str(0)*(6-len(str(gdf1['ID'].iloc[i]))) + str(gdf1['ID'].iloc[i])
        
            with open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/GPS_image/'+ name + '.jpg', 'wb') as f:
                f.write(response.content)
