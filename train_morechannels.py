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



## Parameters
n = 1
experiment_id = '10models_X'
learning_rate = 0.001
patch_size = 64
inchannels = 1
outchannels = 2 
# 0 represents unconfirmed
# 1 represents confirmed as farm
# 2 represents confirmed as not farm
no_epochs = 30
lower_lim = 350
upper_lim = 400
batch_size = 8
continent_no = 2
farm_label_array_path = '/home/kumarv/pravirat/Realsat_labelling/farm_labels.npy'
warped_data_path = '../../WARPED_DATA/350_400_stage2_warped_64x64/'
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
Model_Dir = 'models/test' 
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
        return fc



class CNN_reconstruct(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN_reconstruct, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, 256)
        
        self.upfc = torch.nn.Linear(256, 4096)
        self.unpool3 = torch.nn.ConvTranspose2d(64 , 64, kernel_size=2, stride=2)
        self.upconv3_1 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.upconv3_2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.unpool2 = torch.nn.ConvTranspose2d(32 , 32, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(16 , 16, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(16, in_channels, 3, padding=1)
        
        self.classification_layer = torch.nn.Linear(256, out_channels)

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
farm_conti_list2 = [712882,463811, 465190, 548667, 551790, 553569, 559252, 560828, 602314, 602469, 602694, 603480, 605592, 605723, 606273, 608677, 608988, 611051, 612543, 613360, 614123, 614725, 616005, 618041, 619142, 625020, 631140, 632842, 635446, 635976, 637397, 641082, 651646, 653948, 669396, 669654, 670316, 670765, 670832, 671805, 673381, 676097, 678517, 679522, 685698, 686953, 688942, 691796, 698929, 700459, 702405, 702700, 702997, 703146, 703377, 703563, 703611, 704879, 708867, 709864, 710333, 712944, 712951, 713438, 713610, 713735, 713739, 713768, 714131, 714199, 714290, 714307, 714403, 714674, 714700, 714719, 714738, 714766, 714845, 714865, 714945, 715169, 715193, 715494, 715512, 716086, 719517, 720094, 720177, 720570, 720616, 721237, 723874, 724337, 726925, 729585, 731944, 733586, 734216, 734873, 735854, 736170, 737279, 737622, 738913, 738918, 739287, 739926, 739955, 740186, 713205, 714987, 715048, 715079, 715090, 715164, 715185, 715222, 715284, 715341, 715482, 719586, 720081, 720181, 720845, 720815, 721483, 724549, 727063, 727358, 727488, 729552, 733489, 734286, 734470, 734571, 734672, 734917, 736416, 737561, 738327, 741034, 648880, 636291, 603167, 574180, 601080, 563050, 554086, 550352, 526968, 457312, 720839, 720874, 721609, 723021, 725357, 725599, 726473, 726606, 727403, 727423, 729446, 732465, 734577, 736266, 737469, 738171, 739054, 739381, 741097, 682620, 635989, 616322, 612305, 612180, 611993, 603944, 655242, 709076, 714189, 714427, 714806, 714823, 715080, 715097, 715192, 715226, 722410, 724733, 725175, 725913, 729633, 729634, 738099, 738994, 739390, 601303, 602405, 602773, 641158, 641353, 652423, 667836, 670136, 671536, 671721, 739430, 739163, 738766, 738095, 737526, 725779, 725567, 724502, 715658, 715488, 715131, 715039, 714962, 714345, 714177, 710974, 710930, 703438, 696744, 695299, 693319, 692362]
not_farm_conti_list = []

# get IDs where we know it is farm and subset those poths
farm_IDs = (np.where(farm_label_array == 1)[0]).tolist() # has all the IDs that are farms
not_farm_IDs = (np.where(farm_label_array == 2)[0]).tolist() # has all the IDs that are not farms







for ID in continent_ID_list:
    if ID in not_farm_IDs:
        farm_conti_list2.append(ID)
    if ID in farm_IDs:
        farm_conti_list1.append(ID)
    else:
        not_farm_conti_list.append(ID)

# create the same size of the none figures
no_conti_farm_IDs = len(farm_conti_list1)-len(farm_conti_list2)





## start training 10 different model from here
## so each loop when shuffle pick first several non-farm ID as train sample,
## these IDs will be remove from the not_farm_conti_list
## So for the future loop we will must select totally new IDs as train sample

train_loss = []
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

    not_farm_conti_list = not_farm_conti_list[(len(farm_conti_list1)-len(farm_conti_list2)):]

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
            optimizer.zero_grad()
            classification,out = model(frac_map_batch.to('cuda').float()) # gets output for a batch
            label_batch = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch
            batch_loss_ce = criterion_classification(classification, label_batch) # calculates the loss for that batch
            frac_map_batch = frac_map_batch.to('cuda').float()
            batch_loss_recon = torch.mean(torch.sum(mse_loss(input_image = out, target = frac_map_batch,ignored_index = 0,reduction = 'None')))
            batch_loss = batch_loss_ce + 0.001 * batch_loss_recon
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            epoch_loss_ce += batch_loss_ce.item()
            epoch_loss_recon += batch_loss_recon.item()

        epoch_loss = epoch_loss/(batch+1)
        epoch_loss_ce = epoch_loss_ce/(batch+1)
        epoch_loss_recon = epoch_loss_recon/(batch+1)
        print(epoch_loss,epoch_loss_ce,epoch_loss_recon)
        print('Train Loss:{}  Train Time:{}'.format(epoch_loss, time.time() - train_time_start), end="\n")
        print("\n")
        train_loss.append(epoch_loss)
        # train_loss.append(epoch_loss)
        # train_loss_result = mean(train_loss)


        model.eval()
    #     torch.save(model.state_dict(), os.path.join(Model_Dir, str(experiment_id) + "_epoch_" + str(epoch) + "_train_loss_" + str("{:.4f}".format(train_loss[-1])) +".pt"))
        torch.save(model.state_dict(), os.path.join(Model_Dir, str(experiment_id) +".pt"))

    n = n + 1
    print('reach')
print("TOTAL AVERAGE TRAIN LOSS FOR ",no_epochs,"is ",train_loss_result)
