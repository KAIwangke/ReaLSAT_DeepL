# %%
# Python program to get a google map 
# image of specified location using 
# Google Static Maps API
  
# importing required modules
import requests
  
# Enter your api key here
api_key = "AIzaSyBb9Jp_mE77mAISoOgQY2Fsi3VUr8Iwr8w"
  
# url variable store url
url = "https://maps.googleapis.com/maps/api/staticmap?"
  
# center defines the center of the map,
# equidistant from all edges of the map. 
center = "= "
  
# zoom defines the zoom
# level of the map
zoom = 10
 
# get method of requests module
# return response object
location = '21.644705,109.748304'
# how to get the location


response = requests.get('https://maps.googleapis.com/maps/api/staticmap?center='+str(location)+'&zoom=16&size=640x640&maptype=satellite&key=AIzaSyBb9Jp_mE77mAISoOgQY2Fsi3VUr8Iwr8w')


if response.status_code== 200:
    with open('testFriday1.jpg', 'wb') as f:
        f.write(response.content)

# wb mode is stand for write binary mode
# %%
# sys.path.append("../")
import enum
import os
import numpy as np
import random
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from torch.utils.data.dataset import Dataset
import time
import glob

import os
import torch
import geopandas as gpd
# farm_PATH = '/home/kumarv/pravirat/Realsat_labelling/ASIA_farm/farm_labels_asia.npy'

# farm_PATH_previous = '/home/kumarv/pravirat/Realsat_labelling/farm_labels_store.npy'
# farm_label_array = np.load(farm_PATH)
# paths_list = glob.glob(os.path.join('/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/350_400_stage2_warped_64x64/*.npy'))
# print(len(paths_list))
# path_reservior = "/home/kumarv/pravirat/Realsat_labelling/reservoir_info.npy"
# continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
# continent_no = 1 
# farm_label_array = np.load(farm_PATH)
# newlist = farm_label_array.tolist()

# farms = []
# rivers = []
# for i,d in enumerate(farm_label_array):
#     if d == 1:
#         farms.append(i)
#     if d==2:
#         rivers.append(i)

# print(len(farms))

# farms.remove(21333)
realsat_shape_file_path ="/home/kumarv/pravirat/Realsat_labelling/shape_files/ReaLSAT_351_400_only.shp"

realsat = gpd.read_file(realsat_shape_file_path)

print(realsat)


realsatsub = realsat['geometry']
print(realsatsub[0])

# realsat_subset = realsat[realsat['ID'].isin(farms)]


# print(realsat_subset)

# realsat_subset.to_file("/home/kumarv/pravirat/Realsat_labelling/shape_files/Temp_ReaLSAT_351_400_only_farms.shp", driver = 'ESRI Shapefile')
# %%
