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
print(os.getcwd())

## Parameters
experiment_id = 'test_timeseries'
learning_rate = 0.001
patch_size = 64
inchannels = 1
outchannels = 4 
no_epochs = 50
lower_lim = 350
upper_lim = 400
batch_size = 128
continent_no = 2
mode = 0
Model_Dir = '/home/kumarv/wan00802/models/FourWay_rotateOther/'
continent_no_ASIA = 1
continent_no_US = 2
controlepoch = 100
training = 0
continents = [1,2]
test400 = 0
print("DataSets name: ")
if mode == 0:
    print("Unified Model Generation")
    datapath_test = '/home/kumarv/wan00802/models/ephemeral/datasets/FR_test.npy'
    datapath_test_load = '/home/kumarv/wan00802/models/ephemeral/datasets/FR_test_load.npy'
    datapath_train = '/home/kumarv/wan00802/models/ephemeral/datasets/FR_train.npy'
    datapath_train_load = '/home/kumarv/wan00802/models/ephemeral/datasets/FR_train_load.npy'
elif mode == 1:
    print("Asia Model Generation")
    datapath_test = '/home/kumarv/wan00802/models/ephemeral/datasets/ARAF_test.npy'
    datapath_test_load = '/home/kumarv/wan00802/models/ephemeral/datasets/ARAF_test_load.npy'
    datapath_train = '/home/kumarv/wan00802/models/ephemeral/datasets/ARAF_train.npy'
    datapath_train_load = '/home/kumarv/wan00802/models/ephemeral/datasets/ARAF_train_load.npy'
elif mode == 2:
    print("US Model Generation")
    datapath_test = '/home/kumarv/wan00802/models/ephemeral/datasets/URUF_test.npy'
    datapath_test_load = '/home/kumarv/wan00802/models/ephemeral/datasets/URUF_test_load.npy'
    datapath_train = '/home/kumarv/wan00802/models/ephemeral/datasets/URUF_train.npy'
    datapath_train_load = '/home/kumarv/wan00802/models/ephemeral/datasets/URUF_train_load.npy'




print("continue")
if test400 == 0:
    warped_data_path = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/350_400_stage2_warped_64x64/'
else:
    warped_data_path = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/450_500_stage2_warped_64x64/'
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths

if not os.path.exists(Model_Dir):
    os.makedirs(Model_Dir)


# define deep learning model architecture
class Temp_CNN_simple(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Temp_CNN_simple, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, 256)
        
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

        conv1 = self.maxpool(self.relu(self.conv1_2(self.relu(self.conv1_1(x)))))
        conv2 = self.maxpool(self.relu(self.conv2_2(self.relu(self.conv2_1(conv1)))))
        conv3 = self.maxpool(self.relu(self.conv3_2(self.relu(self.conv3_1(conv2)))))
        conv3 = conv3.view(-1,4096)
        fc = self.fc(conv3)
        
        lstm_enc,_ = self.lstm(t)
        lstm_enc_last = lstm_enc[:,-1]
        
        enc_concat = torch.cat([fc,lstm_enc_last],dim = 1) # concatenating the encodings
        
        out = (self.out(enc_concat))
        
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
        return self.frac_maps[index], self.timeseries[index], self.label_images[index], self.IDs[index]




# testlist=====================================
test_label = np.load(datapath_test)
test_label_load = np.load(datapath_test_load)
print(np.bincount(test_label))
print(np.bincount(test_label_load))
testlist = test_label.tolist()
# exit()
# trainlist=====================================
train_label = np.load(datapath_train)
train_label_load = np.load(datapath_train_load)
print(np.bincount(train_label))
print(np.bincount(train_label_load))
trainlist = train_label.tolist()



TrainFarm = []
TrainRiver = []
trainother=[]
trainE = []
for i,d in enumerate(trainlist):
    if d==1:
        TrainFarm.append(i)
    if d==2:
        TrainRiver.append(i)
    if d == 3:
        trainE.append(i)
    if d == 4:
        trainother.append(i)
        # print(len(trainE))


othertotal = len(trainother)
runcounter = othertotal/len(TrainFarm)
runcounter = int(runcounter)
no_epochs = controlepoch

no_epochs = int(no_epochs)
# training ======================================================

# define loss function
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCELoss()

# build model
model = Temp_CNN_simple(in_channels=inchannels, out_channels=outchannels)
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# get list of paths to use for training 

paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths

print("BUILD MODEL")
model = model.to('cuda')
experiment_id = 0
ACC = 0
ACC_ratio_list = []
thisroundtrainfarm = []
thisroundtrainriver = []
thisroundtrainother = []
total_loss = 0
total_losslist = []
batch_display = 10
num  = 0

continent_info = np.load(continent_info_array_path)
continent_path_list = []
continent_ID_list = []
for path in paths_list:
    ID = path.split('/')[-1].split('_')[-4]
    if(continent_info[int(ID)] == continent_no_ASIA or continent_info[int(ID)]== continent_no_US):
        continent_path_list.append(path)
        ## get all ID from certain continent
        continent_ID_list.append(int(ID))
print("THIS IS CONTINENTS")
print(len(continent_ID_list))


def create_frac_map_and_label(ID,label_array):
#     ID = path.split('/')[-1].split('_')[-4] 
    strID='test'
    if(len(str(ID))!=6):
        complete = 6-len(str(ID))
        strID = '0'*complete+str(ID)
    else:
        strID = str(ID)

    label_image = label_array[int(ID)]
    frac_name = 'ID_'+strID+'_frac_map'+'.npy'
    time_name = 'ID_'+strID+'_time_series'+'.npy'

    frac_map = np.load('/home/kumarv/pravirat/Realsat_labelling/FRAC_MAPS_DATA/350_400_stage2_warped_64x64_frac_map/'+frac_name)
    time_series = np.load('/home/kumarv/pravirat/Realsat_labelling/TIME_SERIES_DATA/350_400_stage2_padded_time_series/'+time_name)
            # print("YES:",ID)
    # image = np.load(warped_data_path + 'ID_' + strID + '_orbit_updated_warped.npy')

    # image[image == 1] = 0 
    # image[image == 2] = 1 
        
    # frac_map_image = np.mean(image,axis = 0)

    # frac_map = np.array(frac_map_image).astype(np.float32)
    # np.save('../temp/'+frac_map_name,frac_map)

    # water_count = []
    # for t in range(image.shape[0]):
    #     water_count.append(np.sum(image[t,0,:,:] == 1))
        
    # time_series = np.array(water_count).astype(np.float32)
    # time_series = time_series/np.max(time_series)
    # time_series = np.expand_dims(time_series, axis=1)
    # np.save('../temp_time/'+frac_map_name,time_series)

    return frac_map, time_series,label_image, ID




tempR  = TrainRiver.copy()
TrainRiver.extend(tempR)
TrainRiver = sample(TrainRiver,len(TrainFarm))

ratio = len(TrainFarm)//len(trainE)
tempE = trainE.copy()        
for i in range(int(ratio)):
    trainE.extend(tempE)
trainE = sample(trainE,len(TrainFarm))


while(training):
    
    FinalIDs_train = []
    # thisroundtrainfarm = sample(TrainFarm,len(TrainRiver))
    thisroundtrainfarm  = TrainFarm.copy()
    # TrainFarm = [x for x in TrainFarm if x not in thisroundtrainfarm]
    
    FinalIDs_train.extend(thisroundtrainfarm)
    print("this is the farms",len(thisroundtrainfarm))

    thisroundtrainriver  = TrainRiver.copy()
    # TrainRiver = [x for x in TrainRiver if x not in thisroundtrainriver]
    FinalIDs_train.extend(thisroundtrainriver)
    print("this is the rivers",len(thisroundtrainriver))


    thisroundtrainE  = trainE.copy()
    # trainE = [x for x in trainE if x not in thisroundtrainE]
    FinalIDs_train.extend(thisroundtrainE)
    print("this is the Ephemerals",len(thisroundtrainE))


    thisroundtrainOther  = sample(trainother,len(TrainFarm))
    trainother = [x for x in trainother if x not in thisroundtrainOther]
    FinalIDs_train.extend(thisroundtrainOther)
    print("this is the others",len(thisroundtrainOther))
    
    print(len(FinalIDs_train))

    
    # print("this is the ephs",len(trainE))

    
    frac_map_images_list_train = []
    timeseries_images_list = []
    label_images_list_train = []
    print(len(FinalIDs_train))
    print("begin to generate the images for train")
# Data loader===============================================================================================
    test = 0
    for ID in FinalIDs_train:
        test +=1
        if test%50 == 0:
            print(test)
        frac_map_image,timeseries_image, label_image, ID = create_frac_map_and_label(ID,train_label_load)            
        frac_map_images_list_train.append(frac_map_image)
        timeseries_images_list.append(timeseries_image)
        label_images_list_train.append(label_image)
    data_train = CLASSIFIER(frac_map_images_list_train,timeseries_images_list, label_images_list_train,FinalIDs_train)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, num_workers=0)
# ===============================================================================================

    if(experiment_id!=0):
        model.load_state_dict(torch.load(os.path.join(Model_Dir, str(experiment_id)+".pt")))
        model = model.to('cuda')
    experiment_id +=1
    model.train()
    print(experiment_id)
    
    for epoch in range(1,no_epochs+1):
        
        for batch, [frac_map_batch, timeseries_batch, label_image_batch,ID_batch] in enumerate(data_loader_train):
            # output = model(frac_map_batch.to('cuda').float())
            target_label = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch
            out = model(frac_map_batch.to('cuda').float(),timeseries_batch.to('cuda').float()) # gets output for a batch
            # print(torch.argmax(out))
            # print(target_label)
            loss = criterion(out, target_label)
            total_loss += loss.item()
            print("EPOCH: ",epoch,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(Model_Dir,str(experiment_id)+'.pt'))


TestFarm = []
TestRiver = []
testother = []
TestE = []
for i,d in enumerate(testlist):
    if d==1:
        TestFarm.append(i)
    if d==2:
        TestRiver.append(i)
    if d==3:
        TestE.append(i)
    if d==4:
        testother.append(i)


test_experiment_id = 3

model.load_state_dict(torch.load(os.path.join(Model_Dir, str(test_experiment_id)+".pt")))
model.eval()

# Data loader===============================================================================================


FinalIDs_test = []
FinalIDs_test.extend(TestFarm)
FinalIDs_test.extend(TestRiver)
FinalIDs_test.extend(TestE)
FinalIDs_test.extend(testother)


print(len(FinalIDs_test))
frac_map_images_list_test = []
timeseries_images_list_test = []

label_images_list_test = []
print("begin to generate the images for test")
# FinalIDs_test = FinalIDs_test[:512]

IDS = FinalIDs_test.copy()

if test400 == 1:
    IDS = continent_ID_list.copy()

for ID in IDS:
    if(len(frac_map_images_list_test)%10==0):
        print(len(frac_map_images_list_test))
    # frac_map_image, label_image, ID = create_frac_map_and_label(ID,test_label_load)   
    frac_map_image,timeseries_image, label_image, ID = create_frac_map_and_label(ID,test_label_load)            
    timeseries_images_list_test.append(timeseries_image)
    frac_map_images_list_test.append(frac_map_image)
    label_images_list_test.append(label_image)
data_test = CLASSIFIER(frac_map_images_list_test,timeseries_images_list_test, label_images_list_test,IDS)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False, num_workers=2)
# ===============================================================================================

preds = []
labels = []
softmax_list =[]
IDs_all = []
for batch, [frac_map_batch,timeseries_batch, label_image_batch,ID_batch] in enumerate(data_loader_test):
    # output = model(frac_map_batch.to('cuda').float()) # gets output for a batch
    target_label = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch
    out = model(frac_map_batch.to('cuda').float(),timeseries_batch.to('cuda').float()) # gets output for a batch

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

listtest = []
for i,d in enumerate(label_array):
    if d ==3 and pred_array[i] == 1:
        listtest.append(IDs_all[i].item())
print(listtest)
print(len(listtest))


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