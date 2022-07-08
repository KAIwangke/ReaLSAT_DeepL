import os
import torch
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys


test_result_10 = []
test_result_9 = []
test_result_8 = []
test_result_7 = []
test_result_6 = []
test_result_5 = []
test_result_4 = []
test_result_3 = []
test_result_2 = []
test_result_1 = []
test_result_0 = []

origin = []

f_10 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level10s.txt','r')
f_9 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level9s.txt','r')
f_8 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level8s.txt','r')
f_7 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level7s.txt','r')
f_6 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level6s.txt','r')
f_5 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level5s.txt','r')
f_4 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level4s.txt','r')
f_3 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level3s.txt','r')
f_2 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level2s.txt','r')
f_1 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level1s.txt','r')
f_0 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level0s.txt','r')

river_label_array = np.load('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/river_labels.npy')
river_label_array = list(river_label_array)


for line in f_10:
    test_result_10.append(int(line.strip()))

for line in f_9:
    test_result_9.append(int(line.strip()))

for line in f_8:
    test_result_8.append(int(line.strip()))

for line in f_7:
    test_result_7.append(int(line.strip()))

for line in f_6:
    test_result_6.append(int(line.strip()))

for line in f_5:
    test_result_5.append(int(line.strip()))

for line in f_4:
    test_result_4.append(int(line.strip()))

for line in f_3:
    test_result_3.append(int(line.strip()))

for line in f_2:
    test_result_2.append(int(line.strip()))

for line in f_1:
    test_result_1.append(int(line.strip()))

for line in f_0:
    test_result_0.append(int(line.strip()))

for i in range(0,len(river_label_array)):
    if river_label_array[i]==1:
        origin.append(i)

## figure out the distribution after removing the labelling river
#%%
for i in origin:
        if i in test_result_10:
            test_result_10.remove(i)
        
        elif i in test_result_9:
            test_result_9.remove(i)

        elif i in test_result_8:
            test_result_8.remove(i)

        elif i in test_result_7:
            test_result_7.remove(i)

        elif i in test_result_6:
            test_result_6.remove(i)

        elif i in test_result_5:
            test_result_5.remove(i)

        elif i in test_result_4:
            test_result_4.remove(i)

        elif i in test_result_3:
            test_result_3.remove(i)

        elif i in test_result_2:
            test_result_2.remove(i)

        elif i in test_result_1:
            test_result_1.remove(i)

        elif i in test_result_0:
            test_result_0.remove(i)


## print the result when exclude the original set
print("origin list:")
print(len(origin))

print("new count == 10:")
print(test_result_10)
print(len(test_result_10))

print("new count == 9:")
print(test_result_9)
print(len(test_result_9))

print("new count == 8:")
print(test_result_8)
print(len(test_result_8))

print("new count == 7:")
print(len(test_result_7))

print("new count == 6:")
print(len(test_result_6))

print("new count == 5:")
print(len(test_result_5))

print("new count == 4:")
print(len(test_result_4))

print("new count == 3:")
print(len(test_result_3))

print("new count == 2:")
print(len(test_result_2))

print("new count == 1:")
print(len(test_result_1))

print("new count == 0:")
print(len(test_result_0))

## check only count == 8 to 10
determine_list = test_result_10 + test_result_9 + test_result_8

## if ID has been already checked as unknown before, remove it
#%%
checked_unknown = []
ff = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/List/unsure.txt','r')
for line in ff:
    checked_unknown.append(int(line.strip()))

for item in determine_list:
    if item in checked_unknown:
        determine_list.remove(item)

#%%
try:
   import config_farm
   print("successfully imported")
except:
    print("not working")

paths_list = glob.glob(os.path.join('../WARPED_DATA/350_400_stage2_warped_64x64/*.npy'))

# print(len(paths_list))

continent_info = np.load('/home/kumarv/pravirat/Realsat_labelling/continent_info.npy')
continent_no = 2 

from matplotlib.backends.backend_pdf import PdfPages

# pp = PdfPages('foo.pdf')
with PdfPages(os.path.join('/panfs/roc/groups/6/kumarv/pravirat/Realsat_labelling/river_code_kx/pdf',"check"+ ".pdf")) as pdf:
    for i,d in enumerate(determine_list):
        ID = determine_list[i]
        for path in paths_list:
        #     print(path)
            ID_path = path.split('/')[-1].split('_')[-4]
            if int(ID) == int(ID_path): 
                fig = plt.figure(figsize = (10,5))
                     
                path_final = path
                # print(ID)
                # print(i)
                image = np.load(path_final)
                print(path_final)
                # converting image to binary
                image[image == 1] = 0
                image[image == 2] = 1

                # take mean over timestamps to create fraction map
                frac_map = np.mean(image,axis = 0)

                water_pixels = []
                for t in range(image.shape[0]):
                    no_water = np.sum(image[t] == 1)
                    water_pixels.append(no_water)

                ax1= fig.add_subplot(1,2,1)
                ax2= fig.add_subplot(1,2,2)

                ax1.plot(water_pixels)
                ax1.title.set_text('ID_'+ str(ID)+ ' water_count')

                ax2.imshow(frac_map[0])
                ax2.title.set_text('ID_'+ str(ID)+ ' Average Image ')

                pdf.savefig(fig)
                fig = plt.figure(figsize=(10, 5))

print("")
print("")
print("")
print("")
#%%

result = []
not_result = []
unknown = []
i = 0
while i < len(determine_list):
    d = input("determine " + str(determine_list[i]) + " is river (1) or not (0) or cannot identify (2)      ")
    if d == '1':
        result.append(determine_list[i])
        i = i + 1
    elif d == '0':
        not_result.append(determine_list[i])
        i = i + 1
    elif d == '2':
        unknown.append(determine_list[i])
        i = i + 1
    else:
        print("Invalid Input, Please reecheck")

print("This is new river")
print(result)
print(len(result))
    
print("This is not river")
print(not_result)
print(len(not_result))

print("This is unknown")
print(unknown)
print(len(unknown))

not_river = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/List/not_river.txt','a')

unsure = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/List/unsure.txt','a')

not_river_list = []
unsure_list = []

for line in not_river_list:
    not_river_list.append(int(line.strip()))


for line in unsure_list:
    unsure_list.append(int(line.strip()))

for item in not_result:
    if item not in not_river_list:
        not_river.write("%s\n" % item)
print('Store Done')
        

for item in unknown:
    if item not in unsure_list:
        unsure.write("%s\n" % item)
print('Store Done')


    


