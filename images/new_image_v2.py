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

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


warped_data_path = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/350_400_stage2_warped_64x64/'

paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths


image_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/GPS_image/EPH/'
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
    image.save(fp="/home/kumarv/pravirat/Realsat_labelling/river_code_kx/GPS_image/EPH/" + name)

print("RESIZE DONE")

## convert png to npy
image_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/GPS_image/EPH/'
image_paths_list = glob.glob(os.path.join(image_path + '*.png'))

for path in image_paths_list:
    img = Image.open(path)
    img = img.convert('RGB')
    ID = path.split('/')[-1].replace('.png','')
    if(len(str(ID))!=6):
        complete = 6-len(str(ID))
        strID = '0'*complete+str(ID)
    else:
        strID = str(ID)
    name = strID +'.npy'
    
    data = np.array( img, dtype='uint8' )

    np.save( '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/GPS_IMAGE_NPY/' + name, data)



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


#labeled_farm = continent_ID_list.copy()

#labeled_farm = [10294, 10778, 30778, 31758, 32005, 33656, 33905, 42260, 46512, 46938, 47095, 51195, 53017, 53338, 54267, 57171, 57361, 57508, 60427, 60646, 60749, 60854, 60940, 61179, 61844, 62339, 64473, 64603, 65106, 65657, 66089, 74497, 75559, 76532, 77993, 85496, 87016, 87619, 87731, 91449, 102623, 103941, 105140, 107835, 108544, 110768, 111209, 112325, 112417, 112574, 113242, 113599, 113898, 116829, 117755, 118376, 118398, 121992, 122643, 123546, 125344, 127720, 129608, 134576, 134623, 135066, 135147, 135999, 136109, 136323, 137958, 138378, 138841, 138951, 139302, 139383, 139556, 139768, 139872, 140771, 141324, 141487, 143216, 144107, 144128, 144524, 144704, 145055, 145689, 145905, 146029, 146281, 151144, 158503, 160527, 161304, 162716, 163946, 164085, 164102, 171982, 172104, 172710, 173784, 174141, 178762, 179364, 180236, 182084, 182256, 182791, 182870, 183257, 183319, 183517, 184197, 186588, 186822, 189721, 191207, 191426, 192091, 194740, 195236, 195560, 195845, 196075, 199839, 200633, 201157, 202171, 202569, 202612, 203151, 203161, 203384, 204166, 204652, 205942, 205946, 206577, 206755, 206864, 207276, 208724, 213535, 219446, 219481, 219692, 220840, 225283, 225293, 226904, 227103, 227365, 227512, 228273, 228357, 228403, 229459, 229734, 230884, 231462, 232864, 232961, 233956, 234164, 235220, 238021, 238446, 238733, 239299, 241468, 241658, 241955, 242327, 243672, 245899, 246061, 249653, 252402, 254086, 254570, 256157, 256229, 256826, 258369, 261543, 261941, 263967, 270110, 270554, 271824, 273128, 274218, 274582, 274598, 274891, 275097, 275940, 276383, 276816, 281233, 281389, 283521, 283628, 284108, 284436, 284766, 285025, 287451, 287516, 288207, 292109, 293017, 293730, 294057, 309606, 309773, 310116, 310251, 311879, 312017, 314127, 314277, 314894, 315161, 315430, 315493, 315984, 316111, 316625, 316689, 316928, 317026, 317817, 317824, 348660, 450433, 451968, 452454, 457664, 458468, 459583, 459668, 459765, 464023, 517858, 520149, 521634, 522033, 523438, 524656, 524970, 526151, 526307, 530544, 531063, 537559, 538586, 538892, 543756, 544451, 545361, 547111, 547133, 547432, 550503, 550780, 551390, 552172, 553468, 554225, 554801, 556043, 556995, 558475, 558832, 559471, 560328, 561378, 562274, 563038, 563222, 563520, 564241, 574417, 601348, 601631, 601652, 601828, 601970, 602032, 602062, 602642, 602807, 602847, 603091, 605588, 605985, 606150, 607144, 607197, 609223, 609849, 610520, 610640, 611097, 615594, 615731, 616174, 616247, 616266, 617460, 617728, 617793, 619168, 619391, 619964, 620244, 620346, 621939, 622708, 623656, 626078, 626417, 628712, 629144, 632076, 634761, 635316, 637076, 637929, 640073, 640599, 648886, 649253, 651028, 651038, 651604, 652251, 653273, 669415, 669729, 670936, 671721, 672069, 675686, 676459, 678409, 679917, 681385, 683345, 683416, 683593, 684548, 685576, 686112, 686264, 690937, 691986, 692274, 692800, 692852, 692939, 694140, 703042, 703065, 703377, 703426, 704785, 708185, 708985, 709161, 710126, 710333, 712882, 712944, 713313, 713419, 713739, 714674, 714766, 715048, 715079, 715169, 715256, 715341, 717124, 717825, 719517, 720081, 720321, 720570, 723267, 724502, 725234, 727075, 729512, 738183, 739407, 739955]

#labeled_farm = [10294, 10778, 30778, 31758, 32005, 33656, 33905, 42260, 46512, 46938, 47095, 51195, 53017, 53338, 54267, 57361, 57508, 60427, 60646, 60749, 60854, 60940, 61179, 61844, 62339, 64473, 64603, 65106, 65657, 66089, 74497, 75559, 76532, 77993, 85496, 87016, 87619, 87731, 91449, 102623, 103941, 105140, 107835, 108544, 110768, 111209, 112325, 112417, 112574, 113242, 113599, 113898, 116829, 118376, 118398, 121992, 122643, 123546, 125344, 127720, 129608, 134576, 134623, 135066, 135147, 135999, 136109, 136323, 137958, 138378, 138841, 138951, 139302, 139383, 139768, 139872, 140771, 141324, 141487, 144107, 144128, 144524, 144704, 145055, 145689, 145905, 146029, 146281, 158503, 160527, 161304, 162716, 163946, 164085, 164102, 172104, 172710, 173784, 174141, 178762, 179364, 180236, 182084, 182256, 182791, 182870, 183319, 183517, 184197, 186588, 186822, 189721, 191207, 191426, 192091, 194740, 195236, 195560, 195845, 196075, 199839, 200633, 201157, 202171, 202569, 202612, 203151, 203161, 203384, 204166, 204652, 205942, 205946, 206577, 206755, 206864, 207276, 208724, 213535, 219481, 219692, 220840, 225283, 225293, 226904, 227103, 227365, 227512, 228273, 228403, 229459, 229734, 230884, 231462, 232864, 232961, 233956, 234164, 235220, 238021, 238733, 239299, 241468, 241658, 241955, 242327, 243672, 245899, 246061, 249653, 252402, 254086, 254570, 256157, 256229, 256826, 258369, 261941, 263967, 270110, 270554, 271824, 273128, 274218, 274582, 274598, 274891, 275097, 275940, 276383, 276816, 281233, 281389, 283521, 283628, 284108, 284436, 284766, 285025, 287451, 287516, 288207, 292109, 293017, 293730, 294057, 309773, 310116, 310251, 311879, 312017, 314127, 314277, 314894, 315161, 315430, 315493, 315984, 316111, 316625, 316928, 317026, 317817, 317824, 348660, 450433, 451968, 452454, 457664, 458468, 459583, 459668, 459765, 464023, 517858, 521634, 522033, 523438, 524656, 524970, 526151, 526307, 530544, 531063, 537559, 538586, 538892, 544451, 545361, 547111, 547432, 550503, 550780, 551390, 552172, 553468, 554225, 554801, 556043, 556995, 558475, 558832, 559471, 560328, 561378, 562274, 563038, 563222, 563520, 564241, 574417, 601348, 601631, 601652, 601828, 601970, 602032, 602062, 602642, 602807, 602847, 603091, 605588, 605985, 606150, 607144, 607197, 609223, 609849, 610640, 611097, 615594, 615731, 616174, 616247, 616266, 617460, 617728, 617793, 619168, 619391, 619964, 620244, 620346, 621939, 622708, 623656, 626078, 626417, 628712, 629144, 632076, 634761, 635316, 637076, 637929, 640073, 640599, 648886, 649253, 651028, 651038, 651604, 652251, 653273, 669415, 669729, 670936, 671721, 672069, 675686, 676459, 678409, 679917, 681385, 683345, 683416, 683593, 684548, 685576, 686112, 686264, 690937, 691986, 692274, 692800, 692852, 692939, 694140, 703042, 703065, 703377, 703426, 704785, 708185, 708985, 709161, 710126, 710333, 712882, 712944, 713313, 713419, 713739, 714674, 714766, 715048, 715079, 715169, 715256, 715341, 717124, 717825, 719517, 720081, 720321, 720570, 723267, 724502, 725234, 727075, 729512, 738183, 739407, 739955]

labeled_farm = [601348, 601652, 601970, 602807, 602847, 620244, 669415, 670936, 710333, 712882, 712944, 713313, 713419, 714674, 715169, 715256, 715341, 727075, 739955, 562513, 563656, 603788, 611600, 611887, 612323, 616322, 616517, 634231, 639371, 639395, 639636, 648804, 648806, 648815, 648963, 680268, 680294, 680725, 682826, 684896, 686689, 687364, 688466, 691781, 691828, 691946, 692031, 692322, 693385, 698974, 702768, 702785, 702929, 703021, 703055, 709076, 710663, 720181, 724350, 725406, 727231, 729945, 730533, 731145, 731925, 732579, 732604, 732651, 732710, 732772, 734216, 734917, 736585, 736616, 736637, 738095, 738171, 738327, 740146, 741060, 741099]

structure = np.ones((3, 3), dtype=np.int)
erosion_structure = np.array([[0,0,0],
                      [0,1,0],
                      [0,0,0]], dtype=np.int)
# exit()


indicator = 0
#with PdfPages(os.path.join('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/400_450_PDF/160*160/',"test"+ ".pdf")) as pdf:
with PdfPages(os.path.join('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/pdf/',"us_ephermal"+ ".pdf")) as pdf:
    for id in labeled_farm:
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
                name = strID +'.npy'

                array = np.load('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/GPS_IMAGE_NPY/'+ name)    
                gpsimg = array_to_img(array)
                #gpsimg = mpimg.imread('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/RESIZE_US_IMAGE/'+ name)
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


# %%


