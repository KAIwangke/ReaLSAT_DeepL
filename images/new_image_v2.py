# %%
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



warped_data_path = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/400_450_stage2_warped_64x64/'

paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths

'''
image_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/US_IMAGE/'
image_paths_list = glob.glob(os.path.join(image_path + '*.png'))

## resize the image
for path in image_paths_list:
    image = Image.open(path)
    ID = path.split('/')[-1].replace('.png','')
    if(len(str(ID))!=6):
        complete = 6-len(str(ID))
        strID = '0'*complete+str(ID)
    else:
        strID = str(ID)
    name = strID +'.png'
    image = image.resize((160,160),Image.ANTIALIAS)
    image.save(fp="/home/kumarv/pravirat/Realsat_labelling/river_code_kx/RESIZE_US_IMAGE/" + name)

'''

continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'

continent_info = np.load(continent_info_array_path)
continent_path_list = []
continent_ID_list = []

continents = [2]

paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths


for path in paths_list:
    ID = path.split('/')[-1].split('_')[-4]
    if(continent_info[int(ID)] in continents):
        continent_path_list.append(path)
        ## get all ID from certain continent
        continent_ID_list.append(int(ID))


labeled_farm = continent_ID_list.copy()

print(len(labeled_farm))
# exit()

structure = np.ones((3, 3), dtype=np.int)
erosion_structure = np.array([[0,0,0],
                      [0,1,0],
                      [0,0,0]], dtype=np.int)

indicator = 0
with PdfPages(os.path.join('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/400_450_PDF/160*160/',"test"+ ".pdf")) as pdf:
    for id in labeled_farm[0:500]:
        ID = id
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

                gpsimg = mpimg.imread('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/RESIZE_US_IMAGE/'+ name)
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


                list = []
                for i in range(0,64):
                    for j in range(0,64):
                        if frac_map[0][i,j] > 0:
                            list.append(frac_map[0][i,j])

                sequence =  sorted(list)
                
                first_quarter =  round(np.percentile(sequence, 25), 2)
                half = round(np.percentile(sequence, 50), 2)
                third_quarter = round(np.percentile(sequence, 75), 2)
            

                ax1= fig.add_subplot(1,7,1)
                ax2= fig.add_subplot(1,7,2)
                ax3= fig.add_subplot(1,7,3)
                ax4= fig.add_subplot(1,7,4)
                ax5= fig.add_subplot(1,7,5)
                ax6= fig.add_subplot(1,7,6)
                ax7= fig.add_subplot(1,7,7)
           

                ax1.plot(water_pixels)
                ax1.title.set_text( str(ID)+ ' water_count'+'Continent: US')

                ax2.imshow(frac_map[0])
                # print(frac_map[0].shape)
                maxindicator = np.max(frac_map[0])
                maxindicator = round(maxindicator, 2) 
                ax2.title.set_text('MAX: '+ str(maxindicator) + ' 25%: '+str(first_quarter) + ' 50%: ' + str(half) + ' 75%: ' + str(third_quarter))

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


