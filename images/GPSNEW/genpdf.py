import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import glob
import numpy as np
import requests
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image
from scipy.ndimage.measurements import label

import pickle




warped_data_path = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/400_450_stage2_warped_64x64/'
paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'



continent_info = np.load(continent_info_array_path)
continent_path_list = []
continent_ID_list = []

continents = [1]
paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths



with open("continent_ID_list", "rb") as fp:   # Unpickling
    continent_ID_list = pickle.load(fp)
print(len(continent_ID_list))


structure = np.ones((3, 3), dtype=np.int)
erosion_structure = np.array([[0,0,0],
                      [0,1,0],
                      [0,0,0]], dtype=np.int)
# exit()
indicator = (6933-len(continent_ID_list))//600

nameodfpdf = str(indicator)+'.pdf'

if (len(continent_ID_list)<600):
    current = continent_ID_list.copy()
else:
    current = continent_ID_list[:600]

continent_ID_list = [x for x in continent_ID_list if x not in current]
with open("continent_ID_list", "wb") as fp:   #Pickling
    pickle.dump(continent_ID_list, fp)
    print(len(continent_ID_list))

labeled_farm = current.copy()

pdf = PdfPages(os.path.join('/panfs/roc/groups/6/kumarv/wan00802/models/result/',""+nameodfpdf))
for i,d in enumerate(labeled_farm):
    ID = labeled_farm[i]
    for path in paths_list:
    #     print(path)
        ID_path = path.split('/')[-1].split('_')[-4]
        if int(ID) == int(ID_path): 
            indicator +=1
            if(indicator%100==0):
                print(indicator)
            fig = plt.figure(figsize = (35,5))
            # gps
            if(len(str(ID))!=6):
                complete = 6-len(str(ID))
                strID = '0'*complete+str(ID)
            else:
                strID = str(ID)
            name = strID +'.png'

            gpsimg = mpimg.imread('/home/kumarv/pravirat/Realsat_labelling/GPSIMAGES/400_450/ASIA_IMAGE/'+name)
            # plt.imshow(img)
            plt.axis('off')     
            
            
            path_final = path
            # print(ID)
            # print(i)
            image = np.load(path_final)
            # print(path_final)
            # converting image to binary
            image[image == 1] = 0
            image[image == 2] = 1

            frac_map = np.mean(image,axis = 0)

            water_pixels = []
            for t in range(image.shape[0]):
                no_water = np.sum(image[t] == 1)
                water_pixels.append(no_water)

            ax1= fig.add_subplot(1,7,1)
            ax2= fig.add_subplot(1,7,2)
            ax3= fig.add_subplot(1,7,3)
            ax4= fig.add_subplot(1,7,4)
            ax5= fig.add_subplot(1,7,5)
            ax6= fig.add_subplot(1,7,6)
            ax7= fig.add_subplot(1,7,7)

            ax1.plot(water_pixels)
            ax1.title.set_text( str(ID)+ ' water_count'+'Continent: ASIA')

            ax2.imshow(frac_map[0])
            # print(frac_map[0].shape)
            maxindicator = np.max(frac_map[0])
            maxindicator = round(maxindicator, 2) 
            ax2.title.set_text(str(ID)+ ' Average Image '+'MAX: '+str(maxindicator))

            ax3.imshow(gpsimg)
            ax3.axis('off') 
            ax3.title.set_text(str(ID)+ ' GPS images ')

            binary_map1 = np.zeros(frac_map[0].shape)
            binary_map1[frac_map[0] >= 0.01] = 1
            count_pos = np.sum(binary_map1 > 0)
            labeled, ncomponents = label(binary_map1, structure)

            ax4.imshow(binary_map1)
            ax4.title.set_text(' Fraction Map 0.01 no.cc:' + str(ncomponents) + ' Count:'+str(count_pos))

            binary_map2 = np.zeros(frac_map[0].shape)
            binary_map2[frac_map[0] >= 0.05] = 1
            count_pos = np.sum(binary_map2 > 0)
            labeled, ncomponents = label(binary_map2, structure)

            ax5.imshow(binary_map2)
            ax5.title.set_text(' Fraction Map 0.05 no.cc:' + str(ncomponents) + ' Count:'+str(count_pos))

            binary_map3 = np.zeros(frac_map[0].shape)
            binary_map3[frac_map[0] >= 0.1] = 1
            count_pos = np.sum(binary_map3 > 0)
            labeled, ncomponents = label(binary_map3, structure)

            ax6.imshow(binary_map3)
            ax6.title.set_text(' Fraction Map 0.1 no.cc:' + str(ncomponents) + ' Count:'+str(count_pos))


            binary_map4 = np.zeros(frac_map[0].shape)
            binary_map4[frac_map[0] >= 0.2] = 1
            count_pos = np.sum(binary_map4 > 0)
            labeled, ncomponents = label(binary_map4, structure)

            ax7.imshow(binary_map4)
            ax7.title.set_text(' Fraction Map 0.2 no.cc:' + str(ncomponents) + ' Count:'+str(count_pos))




            pdf.savefig(fig)
            fig = plt.figure(figsize=(35, 5))
    pdf.close()

