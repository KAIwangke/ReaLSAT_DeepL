# load ID of image of interest
ID = 626849
farm_ID_list = [687369,690423,686539,686326,732351,683799,689417,680826,732094,730702,731630,730576,731431,730612,731328,730504,730545,734435,734214,730304,689616,731761,683693,730997,731571,731631,731813,731723,687981,687846,678214,703809,682868,718233,681844,681656,681524,689334,729234,716578,732657,733084,729152,730212,686979,685421,681652,732494,693355,619367,687214,689523,688805,687528,682984,689234,685028,688435,687666,720406,683774]



labeled_none_farm_stage1 = [418841,463564,520007,547329,540241,622357,693164,558950,573960,554503,463607,414370,519837,5215855,40108,629826]

# i = 0
for i,d in enumerate(labeled_farm):
    ID = labeled_farm[i]
    for path in paths_list:
    #     print(path)
        ID_path = path.split('/')[-1].split('_')[-4]
        if int(ID) == int(ID_path):      
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
            # plotting the fraction map
            fig = plt.figure(figsize=(5,5))
            plt.title(labeled_farm[i])
            plt.imshow(frac_map[0])
            plt.show()
            #count number of water pixels timestamp wise
            water_pixels = []
            for t in range(image.shape[0]):
                no_water = np.sum(image[t] == 1)
                water_pixels.append(no_water)

            # plotting that time series 
            fig = plt.figure(figsize=(5,5))
            plt.title('Water count over time')
            plt.plot(water_pixels)
            plt.show()

# # create timeseries for ID

# # load ID numpy arary
# image = np.load(path_final)

# # converting image to binary
# image[image == 1] = 0
# image[image == 2] = 1

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from torch.utils.data.dataset import Dataset
import time
import glob

## Parameters
experiment_id = 'test'
learning_rate = 0.001
patch_size = 64
inchannels = 1
outchannels = 2
no_epochs = 200
lower_lim = 350
upper_lim = 400
batch_size = 8
continent_no = 2
farm_label_array_path = '/home/kumarv/pravirat/Realsat_labelling/farm_code/farm_labels.npy'
warped_data_path = '../../WARPED_DATA/350_400_stage2_warped_64x64/'
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
Model_Dir = '../models/' + experiment_id

# create model directory if it doesnt exist
if not os.path.exists(Model_Dir):
    os.makedirs(Model_Dir)

# define deep learning model architecture
class CNN_simple(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN_simple, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, out_channels)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax()
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self,x):
        x = x.view(-1, 1, patch_size, patch_size)

        conv1 = self.maxpool(self.relu(self.conv1_2(self.relu(self.conv1_1(x)))))
        conv2 = self.maxpool(self.relu(self.conv2_2(self.relu(self.conv2_1(conv1)))))
        conv3 = self.maxpool(self.relu(self.conv3_2(self.relu(self.conv3_1(conv2)))))
        fc = (self.fc(conv3.view(-1,4096)))
#         fc = self.softmax(fc)
        
        return fc

# define function for creating fraction map and label for a particular ID
def create_frac_map_and_label(ID,label_array):
#     ID = path.split('/')[-1].split('_')[-4]
    image = np.load(warped_data_path + 'ID_' + str(ID) + '_orbit_updated_warped.npy')
    
    # converting labels to binary, i.e land or water
    image[image == 1] = 0
    image[image == 2] = 1
        
    frac_map_image = np.mean(image,axis = 0)
    frac_map = np.array(frac_map_image).astype(np.float32)
    
    label_image = label_array[int(ID)]

    return frac_map, label_image, ID

# create dataloader class
class CLASSIFIER(Dataset):

    def __init__(self, frac_maps, label_images, IDs):
        self.frac_maps = frac_maps
        self.label_images = label_images
        self.IDs = IDs

    def __len__(self):
        return len(self.label_images)

    def __getitem__(self, index):
        return self.frac_maps[index], self.label_images[index],self.IDs[index]

# define loss function
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCELoss()

# build model
print("BUILD MODEL")
model = CNN_simple(in_channels=inchannels, out_channels=outchannels)
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# get list of paths to use for training
farm_label_array = np.load(farm_label_array_path)

paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths

#subset paths that lie in continent of interest
continent_info = np.load(continent_info_array_path)
continent_path_list = []
continent_ID_list = []
for path in paths_list:
    ID = path.split('/')[-1].split('_')[-4]
    if(continent_info[int(ID)] == continent_no):
        continent_path_list.append(path)
        continent_ID_list.append(int(ID))

print(len(continent_path_list))

# get IDs where we know it is farm and subset those poths
farm_IDs = (np.where(farm_label_array == 1)[0]).tolist() # has all the IDs that are farms
print(farm_IDs)
farm_conti_list = []
not_farm_conti_list = []

for ID in continent_ID_list:
    if ID in farm_IDs:
        farm_conti_list.append(ID)
    else:
        not_farm_conti_list.append(ID)
        
print(len(farm_conti_list))
print(len(not_farm_conti_list))
        
no_conti_farm_IDs = len(farm_conti_list)

random.shuffle(not_farm_conti_list)

# store the final IDs into a list
final_IDs = []
for ID in farm_conti_list:
    final_IDs.append(ID)

print(len(final_IDs))
    
for i,ID in enumerate(not_farm_conti_list):
    if(i == no_conti_farm_IDs):
        break
    final_IDs.append(ID)
    
print(len(final_IDs))

random.shuffle(final_IDs)

# exit()

# create fraction maps and labels for all IDs and store in list
frac_map_images_list = []
label_images_list = []

for ID in final_IDs:
    frac_map_image, label_image, ID = create_frac_map_and_label(ID,farm_label_array)
    frac_map_images_list.append(frac_map_image)
    label_images_list.append(label_image)

print(len(label_images_list))

# print(len(frac_map_images_list))
# print(frac_map_images_list[0].shape)

## train model
print("TRAIN MODEL")
train_loss = []

data = CLASSIFIER(frac_map_images_list, label_images_list,final_IDs)
data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=0)

for epoch in range(1,no_epochs+1):
    print('## EPOCH {} ##'.format(epoch))
    model.train()

    train_time_start = time.time()
    epoch_loss = 0
    for batch, [frac_map_batch, label_image_batch,ID_batch] in enumerate(data_loader):
        optimizer.zero_grad()
        out = model(frac_map_batch.to('cuda').float()) # gets output for a batch
        label_batch = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch
        batch_loss = criterion(out, label_batch) # calculates the loss for that batch
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()

    epoch_loss = epoch_loss/(batch+1)
    print('Train Loss:{}  Train Time:{}'.format(epoch_loss, time.time() - train_time_start), end="\n")
    print("\n")
    train_loss.append(epoch_loss)

    model.eval()
#     torch.save(model.state_dict(), os.path.join(Model_Dir, str(experiment_id) + "_epoch_" + str(epoch) + "_train_loss_" + str("{:.4f}".format(train_loss[-1])) +".pt"))
    torch.save(model.state_dict(), os.path.join(Model_Dir, str(experiment_id) +".pt"))
