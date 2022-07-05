#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ast import Not
import enum
from statistics import mean
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

import os
import torch




pdf_path = '/panfs/roc/groups/6/kumarv/pravirat/Realsat_labelling/farm_code_W/reconstruct/reuslt/pdf12'
Model_Dir = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/reconstruct/models/correction2/3channels_version6'
## Parameters
n = 1
experiment_id = '10models_X'
learning_rate = 0.001
patch_size = 64
inchannels = 1
outchannels = 3 
# 0 represents unconfirmed
# 1 represents confirmed as farm
# 2 represents confirmed as not farm
no_epochs = 200
lower_lim = 350
upper_lim = 400
batch_size = 256
continent_no = 2
bias =  0.0000000035
farm_label_array_path = '/home/kumarv/pravirat/Realsat_labelling/farm_labels.npy'
warped_data_path = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/350_400_stage2_warped_64x64/'
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
 
# create model directory if it doesnt exist
if not os.path.exists(Model_Dir):
    os.makedirs(Model_Dir)


if not os.path.exists(pdf_path):
    os.makedirs(pdf_path)
# define deep learning model architecture


class CNN_reconstruct(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN_reconstruct, self).__init__()     
        self.conv1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, 1024)
        
        self.upfc = torch.nn.Linear(1024, 4096)
        self.unpool3 = torch.nn.ConvTranspose2d(64 , 64, kernel_size=2, stride=2)
        self.upconv3_1 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.upconv3_2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.unpool2 = torch.nn.ConvTranspose2d(32 , 32, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(16 , 16, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(16, in_channels, 3, padding=1)
        
        self.classification_layer = torch.nn.Linear(1024, out_channels)
        print(torch.nn.Linear(1024, out_channels))

        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax()
        print(self.softmax)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self,x):
        x = x.view(-1, 1, patch_size, patch_size)

        conv1 = self.maxpool(self.relu(self.conv1_2(self.relu(self.conv1_1(x)))))
        conv2 = self.maxpool(self.relu(self.conv2_2(self.relu(self.conv2_1(conv1)))))
        conv3 = self.maxpool(self.relu(self.conv3_2(self.relu(self.conv3_1(conv2)))))                                     
        fc = self.relu(self.fc(conv3.view(-1,4096)))

        classification = self.classification_layer(fc)

        upfc = self.relu(self.upfc(fc))
        upconv3 = self.relu(self.upconv3_2(self.relu(self.upconv3_1(self.unpool3(upfc.view(-1,64,8,8))))))
        upconv2 = self.relu(self.upconv2_2(self.relu(self.upconv2_1(self.unpool2(upconv3)))))
        out = self.upconv1_2(self.relu(self.upconv1_1(self.unpool1(upconv2))))
        out = out.view(-1, 1, patch_size, patch_size)
        
        return classification,out

def mse_loss(input_image, target, ignored_index, reduction):
    mask = input_image == ignored_index
    out = (input_image[~mask]-target[~mask])**2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out
                                                    
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

criterion_classification = torch.nn.CrossEntropyLoss()


# def testAccuracy():
    
#     model.eval()
#     accuracy = 0.0
#     total = 0.0
    
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             # run the model on the test set to predict labels
#             outputs = model(images)
#             # the label with the highest energy will be our prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             accuracy += (predicted == labels).sum().item()
    
#     # compute the accuracy over all test images
#     accuracy = (100 * accuracy / total)
#     return(accuracy)


# criterion = torch.nn.BCELoss()

# build model
print("BUILD MODEL")
model = CNN_reconstruct(in_channels=inchannels, out_channels=outchannels)
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
        ## get all ID from certain continent
        continent_ID_list.append(int(ID))


farm_conti_list1 = []
farm_conti_list2 = []


torch.cuda.empty_cache()
not_farm_conti_list = []

# get IDs where we know it is farm and subset those poths
farm_IDs = (np.where(farm_label_array == 1)[0]).tolist() # has all the IDs that are farms
not_farm_IDs = (np.where(farm_label_array == 2)[0]).tolist() # has all the IDs that are farms


for ID in continent_ID_list:
    if ID in not_farm_IDs:
        farm_conti_list2.append(ID)

    elif ID in farm_IDs:
        farm_conti_list1.append(ID)
        
    else:
        not_farm_conti_list.append(ID)

# create the same size of the none figures
no_conti_farm_IDs = abs(len(farm_conti_list1)+len(farm_conti_list2))
print(no_conti_farm_IDs)



## start training 10 different model from here
## so each loop when shuffle pick first several non-farm ID as train sample,
## these IDs will be remove from the not_farm_conti_list
## So for the future loop we will must select totally new IDs as train sample


train_loss = []
train_balance = []
for i in range(0,10):

    experiment_id = str(n)
    not_s = []
    random.shuffle(not_farm_conti_list)

    # store the final IDs into a list
    final_IDs = []
    for ID in farm_conti_list1:
        final_IDs.append(ID)

    for ID in farm_conti_list2:
        final_IDs.append(ID)


    for i,ID in enumerate(not_farm_conti_list):
        if(i == no_conti_farm_IDs):
            break
        final_IDs.append(ID)
        not_s.append(ID)

    # print(not_s)

    #no_conti_farm_IDs = no_conti_farm_IDs

    not_farm_conti_list = not_farm_conti_list[(len(farm_conti_list1)+len(farm_conti_list2)):]

    random.shuffle(final_IDs)
    # exit()

    
    # create fraction maps and labels for all IDs and store in list
    frac_map_images_list = []
    label_images_list = []

    for ID in final_IDs:
        frac_map_image, label_image, ID = create_frac_map_and_label(ID,farm_label_array)            
        frac_map_images_list.append(frac_map_image)
        label_images_list.append(label_image)

    print(len(final_IDs))


    # print(len(frac_map_images_list))
    # print(frac_map_images_list[0].shape)

    ## train model
    print("TRAIN MODEL")
    print("MODEL: ",n)


    data = CLASSIFIER(frac_map_images_list, label_images_list,final_IDs)
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=0)
    no_epochs_monitor = 1
    for epoch in range(1,no_epochs+1):
        
        model.train()

        train_time_start = time.time()
        epoch_loss = 0
        epoch_loss_ce = 0
        epoch_loss_recon = 0
    
        for batch, [frac_map_batch, label_image_batch,ID_batch] in enumerate(data_loader):
            classification,out = model(frac_map_batch.to('cuda').float()) # gets output for a batch
            label_batch = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch
            batch_loss_ce = criterion_classification(classification, label_batch) # calculates the loss for that batch
            frac_map_batch = frac_map_batch.to('cuda').float()
            batch_loss_recon = torch.mean(torch.sum(mse_loss(input_image = out, target = frac_map_batch,ignored_index = 0,reduction = 'None')))
            if(no_epochs<130):
                batch_loss = batch_loss_ce
            else:
                batch_loss = bias*batch_loss_recon+batch_loss_ce
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            epoch_loss_ce += batch_loss_ce.item()
            epoch_loss_recon += batch_loss_recon.item()

        epoch_loss = epoch_loss/(batch+1)
        epoch_loss_ce = epoch_loss_ce/(batch+1)
        epoch_loss_recon = epoch_loss_recon/(batch+1)
        epoch_balance = float(epoch_loss_ce/(epoch_loss_recon*bias))
        print("LOSS: ",epoch_loss,epoch_loss_recon)
        print("\n")
        train_loss.append(epoch_loss)
        
        
        # train_loss.append(epoch_loss)
        # train_loss_result = mean(train_loss)

        model.eval()
    #     torch.save(model.state_dict(), os.path.join(Model_Dir, str(experiment_id) + "_epoch_" + str(epoch) + "_train_loss_" + str("{:.4f}".format(train_loss[-1])) +".pt"))
        torch.save(model.state_dict(), os.path.join(Model_Dir, str(experiment_id) +".pt"))
    

    
    n = n + 1
    train_balance.append(epoch_balance)

