#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
import os
import numpy as np
from random import sample
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from torch.utils.data.dataset import Dataset
import time
import glob
import pandas as pd

import psutil

process = psutil.Process(os.getpid())

## Parameters
training = 0
learning_rate = 0.001
patch_size = 64

inchannels = 1
outchannels = 7
no_epochs = 50
lower_lim = 350
upper_lim = 400
batch_size = 64
mode = 0
Model_Dir = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/Model/7way_frac_time/'
continent_no_ASIA = 1
continent_no_US = 2
controlepoch = 100
continents = [1,2]
test400 = 0
print("DataSets name: ")
if mode == 0:
    print("Generation")
    datapath_test = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/DB/new_test.npy'
    datapath_train = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/DB/new_train.npy'
    train_label_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/DB/train_label.npy'
    test_label_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/DB/test_label.npy'


warped_data_path1 = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/350_400_stage2_warped_64x64/'
warped_data_path2 = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/400_450_stage2_warped_64x64/'
warped_data_path3 = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/450_500_stage2_warped_64x64/'


continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
paths_list1 = glob.glob(os.path.join(warped_data_path1 + '*.npy')) # gets all paths
paths_list2 = glob.glob(os.path.join(warped_data_path2 + '*.npy')) # gets all paths
paths_list3 = glob.glob(os.path.join(warped_data_path3 + '*.npy')) # gets all paths

if not os.path.exists(Model_Dir):
    os.makedirs(Model_Dir)

# define deep learning model architecture
class Temp_CNN_simple(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Temp_CNN_simple, self).__init__()
        self.conv1_1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2_1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2_1 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2_1 = torch.nn.Conv2d(64, 64, 3, padding=1)

        self.fc_1_1 = torch.nn.Linear(4096, 256)
 
        self.fc_s = torch.nn.Linear(256,256)
        self.fc_t = torch.nn.Linear(256,256)
    
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=256, batch_first=True)
        
        self.out = torch.nn.Linear(512, out_channels)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax()
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self,x,t):
        x = x.view(-1, 1, patch_size, patch_size)
        conv1 = self.maxpool(self.relu(self.conv1_2_1(self.relu(self.conv1_1_1(x)))))
        conv2 = self.maxpool(self.relu(self.conv2_2_1(self.relu(self.conv2_1_1(conv1)))))
        conv3 = self.maxpool(self.relu(self.conv3_2_1(self.relu(self.conv3_1_1(conv2)))))
        conv3 = conv3.view(-1,4096)
        fc_1 = self.fc_1_1(conv3)
        # print('fc_1')
        # print(fc_1.shape)
        enc_s = self.relu(self.fc_s(fc_1))
        enc_s_norm = torch.nn.functional.normalize(enc_s, p=2.0, dim=1, eps = 1e-12)
        # print('enc_s_norm')
        # print(enc_s_norm.shape)
        # print(enc_gps_norm.shape)
        lstm_enc,_ = self.lstm(t)
        lstm_enc_last = lstm_enc[:,-1]
        enc_t = self.relu(self.fc_t(lstm_enc_last))
        enc_t_norm = torch.nn.functional.normalize(enc_t, p=2.0, dim=1, eps = 1e-12)
     
        # enc_concat_2 = torch.cat([fc_1, fc_2, lstm_enc_last], dim=1) # concatenating the encodings
        # enc_concat_1 = torch.cat([fc_2, lstm_enc_last],dim=1) # concatenating the encodings
        # enc_concat_3 = torch.cat([enc_concat_2, enc_concat_1],dim=1) # concatenating the encodings
  
        # enc_add_1 = torch.add(enc_s_norm, enc_t_norm)
        # enc_add = torch.add(enc_add_1, enc_gps_norm)
        # enc_concat = torch.cat([enc_s_norm, enc_t_norm, enc_gps_norm], dim=1) # concatenating the encodings
        enc_concat = torch.cat([enc_s_norm, enc_t_norm], dim=1) # concatenating the encodings
        # print(enc_concat.shape)
        out = (self.out(enc_concat))
        # print("out")
        # print(out.shape)
        return out

# create dataloader class 
class CLASSIFIER(Dataset):
    def __init__(self, frac_maps, timeseries_IDs, label_images, IDs):
        self.frac_maps = frac_maps
        self.timeseries = timeseries_IDs
        self.label_images = label_images
        self.IDs = IDs

    def __len__(self):
        return len(self.label_images)

    def __getitem__(self, index):
        return self.frac_maps[index], self.timeseries[index],self.label_images[index], self.IDs[index]

def create_frac_map_and_label_v2(ID,label_array):
    strID ='test'
    if(len(str(ID))!=6):
        complete = 6-len(str(ID))
        strID = '0'*complete+str(ID)
    else:
        strID = str(ID)

    label_image = label_array[int(ID)]
    frac_name = 'ID_'+strID+'_frac_map'+'.npy'
    time_name = 'ID_'+strID+'_time_series'+'.npy'
    
    frac_map = np.load('/home/kumarv/pravirat/Realsat_labelling/FRAC_MAPS_DATA/350_500_stage2_warped_64x64_frac_map/'+frac_name)
    time_series = np.load('/home/kumarv/pravirat/Realsat_labelling/TIME_SERIES_DATA/350_500_stage2_padded_time_series/'+time_name)
    return frac_map, time_series, label_image, ID

continent_info = np.load(continent_info_array_path)
continent_ID_list = []

## 350~400
for path in paths_list1:
    ID = path.split('/')[-1].split('_')[-4]
    if(continent_info[int(ID)] == continent_no_ASIA or continent_info[int(ID)]== continent_no_US):
        ## get all ID from certain continent
        continent_ID_list.append(int(ID))

## 400~450
for path in paths_list2:
    ID = path.split('/')[-1].split('_')[-4]
    if(continent_info[int(ID)] == continent_no_ASIA or continent_info[int(ID)]== continent_no_US):
        ## get all ID from certain continent
        continent_ID_list.append(int(ID))

## 450~500
for path in paths_list3:
    ID = path.split('/')[-1].split('_')[-4]
    if(continent_info[int(ID)] == continent_no_ASIA or continent_info[int(ID)]== continent_no_US):
        ## get all ID from certain continent
        continent_ID_list.append(int(ID))

print("THIS IS CONTINENTS")
print(len(continent_ID_list))

# testlist=====================================
train_label = np.load(train_label_path)
test_list = np.load(datapath_test)
print(np.bincount(test_list))
testlist = test_list.tolist()

# trainlist=====================================
test_label = np.load(test_label_path)
train_list = np.load(datapath_train)
print(np.bincount(train_list))
trainlist = train_list.tolist()

TrainFarm = []
TrainReservoir = []
TrainRiver = []
TrainNatural=[]
TrainSeasonal = []
TrainHigh = []
TrainEph = []
for i,d in enumerate(trainlist):
    if d == 1:
        TrainFarm.append(i)
    if d == 2:
        TrainReservoir.append(i)
    if d == 3:
        TrainRiver.append(i)
    if d == 4:
        TrainNatural.append(i)
    if d == 5:
        TrainSeasonal.append(i)
    if d == 6:
        TrainHigh.append(i)
    if d == 7:
        TrainEph.append(i)
        # print(len(trainE))

no_epochs = controlepoch
# training ======================================================


criterion = torch.nn.CrossEntropyLoss()
model = Temp_CNN_simple(in_channels=inchannels, out_channels=outchannels)
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("BUILD MODEL")
model = model.to('cuda')
experiment_id = 0
ACC = 0
ACC_ratio_list = []

thisroundtrainfarm = []
thisroundtrainreservoir = []
thisroundtrainriver = []
thisroundtrainnatural = []
thisroundtrainseason = []
thisroundtrainhigh = []
thisroundtraineph = []

total_loss = 0
total_losslist = []
batch_display = 10
num  = 0

while(training):
    thisround_train = []

    thisroundtrainfarm  = sample(TrainFarm, len(TrainEph))
    TrainFarm = [x for x in TrainFarm if x not in thisroundtrainfarm]
    print("this is the farms", len(thisroundtrainfarm))

    thisroundtrainreservoir  = sample(TrainReservoir, len(TrainEph))
    print("this is the reservoir", len(thisroundtrainfarm))
    
    thisroundtrainriver  = sample(TrainRiver, len(TrainEph))
    print("this is the rivers", len(thisroundtrainriver))
 
    ratio1 = int(len(TrainEph)//len(TrainNatural))
    for i in range(ratio1):
        thisroundtrainnatural.extend(TrainNatural)
    print("this is the natural lakes", len(thisroundtrainnatural))
    
    thisroundtrainseason  = sample(TrainSeasonal, len(TrainEph))
    print("this is the seasonal lakes", len(thisroundtrainseason))

    ratio2 = int(len(TrainEph)//len(TrainHigh))
    for i in range(ratio2):
        thisroundtrainhigh.extend(TrainHigh)
    print("this is the high-seasonal lakes", len(thisroundtrainhigh))
  
    thisroundtraineph = TrainEph
    print("this is the Ephemerals",len(thisroundtraineph))

    thisround_train.extend(thisroundtrainfarm)
    thisround_train.extend(thisroundtrainreservoir)
    thisround_train.extend(thisroundtrainriver)
    thisround_train.extend(thisroundtrainnatural)
    thisround_train.extend(thisroundtrainseason)
    thisround_train.extend(thisroundtrainhigh)
    thisround_train.extend(thisroundtraineph)
    
    # print("this is the ephs",len(trainE))
    
    frac_map_images_list_train = []
    timeseries_images_list = []
    label_images_list_train = []
    print(len(thisround_train))
    print("begin to generate the images for train")

# Data loader===============================================================================================
    test = 0
    for ID in thisround_train:
        test +=1
        if test%200 == 0:
            print(test)
        frac_map_image, timeseries_image, label_image, ID = create_frac_map_and_label_v2(ID,train_label)     
        # print(timeseries_image.shape)
        # exit()
        # print(frac_map_image.shape)

        frac_map_images_list_train.append(frac_map_image)
        timeseries_images_list.append(timeseries_image)
        label_images_list_train.append(label_image)

    data_train = CLASSIFIER(frac_map_images_list_train, timeseries_images_list,label_images_list_train, thisround_train)

    
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, num_workers=0)

# ===============================================================================================

    if(experiment_id!=0):
        model.load_state_dict(torch.load(os.path.join(Model_Dir, str(experiment_id)+".pt")))
        model = model.to('cuda')
    experiment_id +=1
    model.train()  # it just set the mode

    print(experiment_id)

    for epoch in range(1,no_epochs+1):
        for batch, [frac_map_batch, timeseries_batch, label_image_batch, ID_batch] in enumerate(data_loader_train):
            # output = model(frac_map_batch.to('cuda').float())
            target_label = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch
            # print(frac_map_batch.size())
            # print(gps_image_batch.size())
            # print(timeseries_batch.size())
            # exit()
            out = model(frac_map_batch.to('cuda').float(),timeseries_batch.to('cuda').float()) # gets output for a batch
      
            # print(torch.argmax(out))
            loss = criterion(out, target_label)
            total_loss += loss.item()
            print("EPOCH: ",epoch,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(Model_Dir, str(experiment_id)+'.pt'))

test_experiment_id = 3

model.load_state_dict(torch.load(os.path.join(Model_Dir, str(test_experiment_id)+".pt")))
model.eval()

# Data loader===============================================================================================

test = []
for i in range(len(testlist)):
    if testlist[i] == 1:
        test.append(i)
    if testlist[i] == 2:
        test.append(i)
    if testlist[i] == 3:
        test.append(i)
    if testlist[i] == 4:
        test.append(i)
    if testlist[i] == 5:
        test.append(i)
    if testlist[i] == 6:
        test.append(i)
    if testlist[i] == 7:
        test.append(i)

print(len(test))
frac_map_images_list_test = []
timeseries_images_list_test = []

label_images_list_test = []
print("begin to generate the images for test")
# FinalIDs_test = FinalIDs_test[:512]

IDS = test.copy()

if test400 == 1:
    IDS = continent_ID_list.copy()

for ID in IDS:
    if(len(frac_map_images_list_test)%10==0):
        print(len(frac_map_images_list_test))
    # frac_map_image, label_image, ID = create_frac_map_and_label(ID,test_label_load)   
    frac_map_image, timeseries_image, label_image, ID = create_frac_map_and_label_v2(ID,test_label)            
    timeseries_images_list_test.append(timeseries_image)
    frac_map_images_list_test.append(frac_map_image)
    label_images_list_test.append(label_image)
data_test = CLASSIFIER(frac_map_images_list_test,timeseries_images_list_test ,label_images_list_test,IDS)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False, num_workers=2)

# ===============================================================================================

preds = []
labels = []
softmax_list =[]
IDs_all = []

for batch, [frac_map_batch,timeseries_batch, label_image_batch, ID_batch] in enumerate(data_loader_test):
    # output = model(frac_map_batch.to('cuda').float()) # gets output for a batch
    target_label = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch
    out = model(frac_map_batch.to('cuda').float(), timeseries_batch.to('cuda').float()) # gets output for a batch

    softmax_batch = torch.nn.functional.softmax(out, dim=1)
    ID_batch_cpu = ID_batch.detach().cpu().numpy()
    out_label_batch = torch.argmax(softmax_batch, dim=1)
    out_label_batch_cpu = out_label_batch.detach().cpu().numpy()
    label_batch_cpu = target_label.detach().cpu().numpy()

    softmax_batch_cpu = softmax_batch.detach().cpu().numpy()
    for b in range(softmax_batch_cpu.shape[0]):
        softmax_list.append(softmax_batch_cpu[b])
        IDs_all.append(ID_batch_cpu[b])
        preds.append(out_label_batch_cpu[b])
        labels.append(label_batch_cpu[b])


    
pred_array = np.array(preds)
label_array = np.array(labels)
softmax_list = np.array(softmax_list)
farm_score = softmax_list[:,1]
river_score = softmax_list[:,2]
eph_score = softmax_list[:,3]


np.save("predictionlist",pred_array)
np.save("manualableed",label_array)
ephs = []

if test400 == 1:
    for i,d in enumerate(eph_score):
        if d>=0.95:
            ephs.append(IDs_all[i].item())
    print(ephs)
    print(len(ephs))

print(pred_array.shape)
print(label_array.shape)

print("np.bincount(pred_array) ",np.bincount(pred_array))
print("np.bincount(label_array))",np.bincount(label_array))
print(classification_report(label_array, pred_array, digits=4))
print(confusion_matrix(label_array,pred_array))

listtest1 = []
for i,d in enumerate(label_array):
    if d ==6 and pred_array[i] == 0:
        listtest1.append(IDs_all[i].item())

print("These are actual ephermal but predicted as farm")
print(listtest1)
print(len(listtest1))

listtest2 = []
for i,d in enumerate(label_array):
    if d == 0 and pred_array[i] == 6:
        listtest2.append(IDs_all[i].item())

print("These are actual farm but predicted as ephermal")
print(listtest2)
print(len(listtest2))

listtest3 = []
for i,d in enumerate(label_array):
    if d == 1 and pred_array[i] == 0:
        listtest3.append(IDs_all[i].item())

print("These are actual reservoir but predicted as farm")
print(listtest3)
print(len(listtest3))

print(f1_score(y_true=label_array, y_pred=pred_array, average='macro'))

# softmax_list = np.array(softmax_list)
# other_score = softmax_list[:,0]
# farm_score = softmax_list[:,1]
# river_score = softmax_list[:,2]
# eph_score = softmax_list[:,3]


# fig = plt.figure(figsize = (10,30))

# ax1= fig.add_subplot(3,1,1)
# ax2= fig.add_subplot(3,1,2)
# ax3= fig.add_subplot(3,1,3)
# counts,edges,bars = ax1.hist(other_score,bins = np.linspace(0,1,num=11))
# ax1.bar_label(bars)
# ax1.title.set_text('Others histogram')
# counts,edges,bars = ax2.hist(farm_score,bins = np.linspace(0,1,num=11))
# ax2.bar_label(bars)
# ax2.title.set_text('Farm histogram')
# counts,edges,bars = ax3.hist(river_score,bins = np.linspace(0,1,num=11))
# ax3.bar_label(bars)
# ax3.title.set_text('River histogram')

# plt.savefig('/home/kumarv/wan00802/models/result/softmax_LSTM_CNN.png')
