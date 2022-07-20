from ast import If, Not
import enum
from statistics import mean
import sys
from random import sample



# from farm_code_W.reconstruct.train.Iteration_train import Accuracy
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
n=1
Accuracy = 0
bias =  0.0000000065
# bias =  0.0000000175
no_epochs = 100
trainmodelname ='65_100_addlength_us'
Model_Dir = '/home/kumarv/pravirat/Realsat_labelling/UnifiedModel/Models/65_100_addlength_us/'
## Parameters
experiment_id = '10models_X'
learning_rate = 0.001
patch_size = 64
inchannels = 1
outchannels = 3 
# 0 represents unconfirmed
# 1 represents confirmed as farm
# 2 represents confirmed as not farm

lower_lim = 350
upper_lim = 400
batch_size = 512
continent_no_ASIA = 1
continent_no_US = 2

datapath_test = '/home/kumarv/pravirat/Realsat_labelling/UnifiedModel/new_datasets/URUF_test.npy'
datapath_test_load = '/home/kumarv/pravirat/Realsat_labelling/UnifiedModel/new_datasets/URUF_test_load.npy'
datapath_train = '/home/kumarv/pravirat/Realsat_labelling/UnifiedModel/new_datasets/URUF_train.npy'
datapath_train_load = '/home/kumarv/pravirat/Realsat_labelling/UnifiedModel/new_datasets/URUF_train_load.npy'

warped_data_path = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/350_400_stage2_warped_64x64/'
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths

if not os.path.exists(Model_Dir):
    os.makedirs(Model_Dir)

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
    strID='test'
    if(len(str(ID))!=6):
        complete = 6-len(str(ID))
        strID = '0'*complete+str(ID)
    else:
        strID = str(ID)
    image = np.load(warped_data_path + 'ID_' + strID + '_orbit_updated_warped.npy')
    
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

continent_info = np.load(continent_info_array_path)
continent_path_list = []
continent_ID_list = []
for path in paths_list:
    ID = path.split('/')[-1].split('_')[-4]
    if(continent_info[int(ID)] == continent_no_ASIA or continent_info[int(ID)]== continent_no_US):
        continent_path_list.append(path)
        ## get all ID from certain continent
        continent_ID_list.append(int(ID))




# testlist=====================================
test_label = np.load(datapath_test)
test_label_load = np.load(datapath_test_load)
testlist = test_label.tolist()
# exit()
# trainlist=====================================
train_label = np.load(datapath_train)
train_label_load = np.load(datapath_train_load)
trainlist = train_label.tolist()

TestFarm = []
TestRiver = []
TestFR = []
testother = []


for i,d in enumerate(testlist):
    if d==1:
        TestFarm.append(i)
        TestFR.append(i)
    if d==2:
        TestRiver.append(i)
        TestFR.append(i)
    if d==3:
        testother.append(i)


print(len(testother))

TrainFarm = []
TrainRiver = []
TrainFR = []
trainother=[]
for i,d in enumerate(trainlist):
    if d==1:
        TrainFarm.append(i)
        TrainFR.append(i)
    if d==2:
        TrainRiver.append(i)
        TrainFR.append(i)
    if d == 3:
        trainother.append(i)

print(len(trainother))

# Data loader===============================================================================================


FinalIDs_test = []
FinalIDs_test.extend(TestFarm)
FinalIDs_test.extend(TestRiver)
FinalIDs_test.extend(testother)
frac_map_images_list_test = []
label_images_list_test = []
print(len(FinalIDs_test))
print("begin to generate the images for test")
# FinalIDs_test = FinalIDs_test[:512]


for ID in FinalIDs_test:
    if(len(frac_map_images_list_test)%1000==0):
        print(len(frac_map_images_list_test))
    frac_map_image, label_image, ID = create_frac_map_and_label(ID,test_label_load)            
    frac_map_images_list_test.append(frac_map_image)
    label_images_list_test.append(label_image)
data_test = CLASSIFIER(frac_map_images_list_test, label_images_list_test,FinalIDs_test)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=0)
# ===============================================================================================

# build model
print("BUILD MODEL")
model = CNN_reconstruct(in_channels=inchannels, out_channels=outchannels)
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


experiment_id = 0
ACC = 0
ACC_ratio_list = []
thisroundtrainfarm = []
thisroundtrainriver = []
thisroundtrainother = []
while(ACC<0.95):
    FinalIDs_train = []
    thisroundtrainfarm = sample(TrainFarm,len(TrainRiver))

    FinalIDs_train.extend(thisroundtrainfarm)
    FinalIDs_train.extend(TrainRiver)
    trainother = [x for x in trainother if x not in thisroundtrainother]
    otherlen = len(TrainRiver)*2
    if(len(trainother)<otherlen):
        print("finished training round")
        break
    thisroundtrainother = sample(trainother,otherlen)
    FinalIDs_train.extend(thisroundtrainother)
    frac_map_images_list_train = []
    label_images_list_train = []
    print(len(FinalIDs_train))
    print("begin to generate the images for train")
    # FinalIDs_train = FinalIDs_train[:512]

    for ID in FinalIDs_train:
        frac_map_image, label_image, ID = create_frac_map_and_label(ID,train_label_load)            
        frac_map_images_list_train.append(frac_map_image)
        label_images_list_train.append(label_image)
    data_train = CLASSIFIER(frac_map_images_list_train, label_images_list_train,FinalIDs_train)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, num_workers=0)

    # training ======================================================================================================
    if(experiment_id!=0):
        model.load_state_dict(torch.load(os.path.join(Model_Dir, str(experiment_id)+".pt")))
        model = model.to('cuda')
    experiment_id +=1
    model.train()
    print(experiment_id)
        
    for epoch in range(1,no_epochs+1):
        if(epoch%10 == 0):
            print(epoch)
        for batch, [frac_map_batch, label_image_batch,ID_batch] in enumerate(data_loader_train):
            classification,out = model(frac_map_batch.to('cuda').float()) # gets output for a batch
            label_batch = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch

            batch_loss_ce = criterion_classification(classification, label_batch) # calculates the loss for that batch
            frac_map_batch = frac_map_batch.to('cuda').float()
            batch_loss_recon = torch.mean(torch.sum(mse_loss(input_image = out, target = frac_map_batch,ignored_index = 0,reduction = 'None')))
            # if(no_epochs<(no_epochs*3/4)):
            #     batch_loss = batch_loss_ce
            # else:
            batch_loss = bias*batch_loss_recon+batch_loss_ce
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        
        # Testing ======================================================================================================
        if(epoch%no_epochs == 0):
            # print(epoch,"LOSS: ",epoch_loss,epoch_loss_recon)
            model.eval()
            preds = []
            labels = []
            IDs_all = []
            totallist1= test_label.tolist()
            for i in range(1000000):
                totallist1[i] = 0
            totallist2 = totallist1.copy()
            for batch, [frac_map_batch, label_image_batch,ID_batch] in enumerate(data_loader_test):
                classification,out = model(frac_map_batch.to('cuda').float()) # gets output for a batch
                label_batch = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch
                out_label_batch = torch.argmax(torch.nn.functional.softmax(classification, dim=1), dim=1)
                preds  = list(preds)
                preds.append(out_label_batch.detach().cpu().numpy())
                labels.append(label_batch.cpu().numpy())
                IDs_all.append(ID_batch)
            pred_array = np.concatenate(preds, axis=0)
            ID_array = np.concatenate(IDs_all, axis=0)
            # print(classification_report(ID_array, pred_array, digits=4))
            # print(f1_score(y_true=ID_array, y_pred=pred_array, average='macro'))

            for i,d in enumerate(ID_array):
                if pred_array[i] == 1:
                    totallist1[d] += pred_array[i]  
                if pred_array[i] == 2:
                    totallist2[d] +=1    
            farm_prediction=[]
            counting1 = 0          

            for i,d in enumerate(totallist1):
                if(d == 1):
                    counting1+=1
                    farm_prediction.append(i)
            river_prediction=[]
            counting2 = 0
            for i,d in enumerate(totallist2):
                if(d == 1):
                    counting2+=1
                    river_prediction.append(i)

            farm_overlap = [x for x in TestFarm if x in farm_prediction]
            print(counting1," FARM ACCURACY: ",len(farm_overlap)/len(farm_prediction))

            river_overlap = [x for x in TestRiver if x in river_prediction]
            print(counting2," RIVER ACCURACY: ",len(river_overlap)/len(river_prediction))
            
            torch.save(model.state_dict(), os.path.join(Model_Dir, str(experiment_id) +".pt"))
            
            ACC = (len(farm_overlap)+len(river_overlap))/(len(farm_prediction)+len(river_prediction))
            ACC_ratio_list.append(ACC)
            print(max(ACC_ratio_list))
torch.save(model.state_dict(), os.path.join(Model_Dir,trainmodelname))