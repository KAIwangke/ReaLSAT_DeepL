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
no_epochs = 50
lower_lim = 350
upper_lim = 400
batch_size = 8
continent_no = 2
farm_label_array_path = '/home/kumarv/pravirat/Realsat_labelling/farm_code_kx/farm_labels.npy'
warped_data_path = '../../WARPED_DATA/350_400_stage2_warped_64x64/'
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
Model_Dir = '../models/' + experiment_id
model_name = experiment_id + '.pt'

if not os.path.exists(Model_Dir):
    os.makedirs(Model_Dir)


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

class CLASSIFIER(Dataset):

    def __init__(self, frac_maps, label_images, IDs):
        self.frac_maps = frac_maps
        self.label_images = label_images
        self.IDs = IDs

    def __len__(self):
        return len(self.label_images)

    def __getitem__(self, index):
        return self.frac_maps[index], self.label_images[index],self.IDs[index]


criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCELoss()

farm_ID_list = [687369,690423,686539,686326,732351,683799,689417,680826,732094,730702,731630,730576,731431,730612,731328,730504,730545,734435,734214,730304,689616,731761,683693,730997,731571,731631,731813,731723,687981,687846,678214,703809,682868,718233,681844,681656,681524,689334,729234,716578,732657,733084,729152,730212,686979,685421,681652,732494,693355,619367,687214,689523,688805,687528,682984,689234,685028,688435,687666,720406,683774]

# %%
print("BUILD MODEL")
model = CNN_simple(in_channels=inchannels, out_channels=outchannels)
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("LOAD MODEL")
print(model_name)
model.load_state_dict(torch.load(os.path.join(Model_Dir, model_name)))
model.eval()

# %%
print("#######################################################################")
print("TEST MODEL")
count = 1
while len(farm_ID_list) <= 100:
    
    print("ITERATION" + " " + str(count))

    train_loss = []

    image_patches_list = []
    label_patches_list = []
    farm_label_array = np.load(farm_label_array_path)

    # paths_list = glob.glob(os.path.join(config.PAD_DIR,config.PAD_FOLDER, '*.npy'))
    paths_list = glob.glob(os.path.join(warped_data_path + '*.npy'))

    continent_info = np.load(continent_info_array_path)
    continent_path_list = []
    continent_ID_list = []
    for path in paths_list:
        ID = path.split('/')[-1].split('_')[-4]
        if(continent_info[int(ID)] == continent_no):
            continent_path_list.append(path)
            continent_ID_list.append(int(ID))

    ##print(len(continent_path_list))

    farm_IDs = (np.where(farm_label_array == 1)[0]).tolist()
    farm_conti_list = []
    not_farm_conti_list = []

    for ID in continent_ID_list:
        if ID in farm_IDs:
            farm_conti_list.append(ID)
        else:
            not_farm_conti_list.append(ID)
            
    no_conti_farm_IDs = len(farm_conti_list)

    random.shuffle(not_farm_conti_list)

    final_IDs = []

    for ID in farm_conti_list:
        final_IDs.append(ID)
        
    for i,ID in enumerate(not_farm_conti_list):
        if(i == no_conti_farm_IDs):
            break
        final_IDs.append(ID)

    random.shuffle(final_IDs)

    frac_map_images_list = []
    label_images_list = []

    for ID in final_IDs:
        frac_map_image, label_image, ID = create_frac_map_and_label(ID,farm_label_array)            
        frac_map_images_list.append(frac_map_image)
        label_images_list.append(label_image)

    print(len(label_images_list))

    data = CLASSIFIER(frac_map_images_list, label_images_list,final_IDs)
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=0)


    test_time_start = time.time()
    loss = 0
    preds = []
    labels = []
    IDs_all = []
    for batch, [frac_map_batch, label_image_batch,ID_batch] in enumerate(data_loader):
        optimizer.zero_grad()
        out = model(frac_map_batch.to('cuda').float())
        label_batch = label_image_batch.type(torch.long).to('cuda')
        batch_loss = criterion(out, label_batch)
    #     print(out)
    #     print(torch.nn.functional.softmax(out, dim=1))
    #     print(torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1))
    #     print(label_batch)
        
        out_label_batch = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
    #     exit()
        loss += batch_loss.item()
        
        preds.append(out_label_batch.detach().cpu().numpy())
        labels.append(label_batch.cpu().numpy())
        IDs_all.append(ID_batch)

    loss = loss/(batch+1)
    print('Test Loss:{}  Test Time:{}'.format(loss, time.time() - test_time_start), end="\n")
    print("\n")

    pred_array = np.concatenate(preds, axis=0)
    label_array = np.concatenate(labels, axis=0)
    ID_array = np.concatenate(IDs_all, axis=0)
    
    '''
    print(pred_array.shape)
    print(label_array.shape)
    '''

    print(classification_report(label_array, pred_array, digits=4))
    print(f1_score(y_true=label_array, y_pred=pred_array, average='macro'))

    
    print(pred_array)
    print("#####################")
    print(label_array)
    print("######################")
    print(ID_array)
       
    print( type(pred_array[0]))
    
    print( type(label_array[0]))
    
    for i in range(0,len(pred_array)):
        if int(pred_array[i]) == 1:
            if pred_array[i] != label_array[i]:
                if ID_array[i] in farm_ID_list:
                    continue
                else:
                    farm_ID_list.append(ID_array[i])
            else:
                continue
        elif int(pred_array[i]) == 0:
            if pred_array[i] != label_array[i]:
                if ID_array[i] in farm_ID_list:
                    farm_ID_list.remove(ID_array[i])
                else:
                    continue
            else:
                continue
        
    print(len(farm_ID_list))
    count = count + 1

print(farm_ID_list)

'''
farm_label_array = np.load('/home/kumarv/pravirat/Realsat_labelling/farm_code_kx/farm_labels.npy')
for farm_ID in farm_ID_list:
    farm_label_array[int(farm_ID)] = 1

np.save('/home/kumarv/pravirat/Realsat_labelling/farm_code_kx/farm_labels.npy',farm_label_array)
'''


# print(pred_array)
# print(label_array)
# print(ID_array)

# for i in range(label_array.shape[0]):
#     if(pred_array[i] == 1 and label_array[i] == 0):
#         print(ID_array[i])
