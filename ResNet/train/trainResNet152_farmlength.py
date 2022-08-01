from ast import If, Not
import enum
from statistics import mean
import sys
from random import sample
import math
import cv2

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
import cv2
import os
import torch
import pandas as pd



# =============================================================================================================
n=1
training = 0
# 1 -> only training without testing
# 0 -> only testing without training
mode = 0
controlepoch = 500
# 0 -> Unified Model
# 1 -> Asia Model
# 2 -> US Model


Accuracy = 0

Model_Dir = '/home/kumarv/wan00802/models/resnet152_500epochs/'
print("Model name: "+Model_Dir)
## Parameters
learning_rate = 0.0001
patch_size = 224
# inchannels = 3 default input image size is (224,224,3)
in_channels = 1
outchannels = 3 
# 0 represents unconfirmed
# 1 represents confirmed as farm
# 2 represents confirmed as not farm

lower_lim = 350
upper_lim = 400
batch_size = 64
continent_no_ASIA = 1
continent_no_US = 2
# =============================================================================================================

print("DataSets name: ")
if mode == 0:
    print("Unified Model Generation")
    datapath_test = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/FR_test.npy'
    datapath_test_load = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/FR_test_load.npy'
    datapath_train = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/FR_train.npy'
    datapath_train_load = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/FR_train_load.npy'
elif mode == 1:
    print("Asia Model Generation")
    datapath_test = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/ARAF_test.npy'
    datapath_test_load = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/ARAF_test_load.npy'
    datapath_train = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/ARAF_train.npy'
    datapath_train_load = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/ARAF_train_load.npy'
elif mode == 2:
    print("US Model Generation")
    datapath_test = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/URUF_test.npy'
    datapath_test_load = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/URUF_test_load.npy'
    datapath_train = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/URUF_train.npy'
    datapath_train_load = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/URUF_train_load.npy'



print("continue")

warped_data_path = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/350_400_stage2_warped_64x64/'
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths

if not os.path.exists(Model_Dir):
    os.makedirs(Model_Dir)

class ResNet(torch.nn.Module):
    def __init__(self, layers,num_classes = outchannels):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # self.conv1 = torch.nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3,bias = False)
        self.conv1 = torch.nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace = True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.stack1 = self.make_stack(64, layers[0])
        self.stack2 = self.make_stack(128, layers[1], stride=2)
        self.stack3 = self.make_stack(256, layers[2], stride=2)
        self.stack4 = self.make_stack(512, layers[3], stride=2)
        self.avgpool = torch.nn.AvgPool2d(3, stride = 1)

        self.fc = torch.nn.Linear(2048 * Bottleneck.expansion, num_classes)
        # initialize parameters
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2./n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride = 1):
        downsample = None
        layers = []
            
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * Bottleneck.expansion),
                )

        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * 4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # important=====
        out += residual #=
        # ===============
        out = self.relu(out)

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

    label_image = label_array[int(ID)]
    frac_map_name = str(ID)+'.npy'
    if os.path.exists('../temp/'+frac_map_name):
        frac_map = np.load('../temp/'+frac_map_name)
    else:
        image = np.load(warped_data_path + 'ID_' + strID + '_orbit_updated_warped.npy')
        
        # converting labels to binary, i.e land or water
        image[image == 1] = 0 
        image[image == 2] = 1 
            
        frac_map_image = np.mean(image,axis = 0)

        frac_map = np.array(frac_map_image).astype(np.float32)
        np.save('../temp/'+frac_map_name,frac_map)

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
criterion = torch.nn.CrossEntropyLoss().cuda()



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
othertotal = len(trainother)
runcounter = othertotal/len(TrainFarm)
runcounter = int(runcounter)
no_epochs = controlepoch//runcounter

no_epochs = int(no_epochs)
print("number of epochs in total will be ",no_epochs)
print("number of runs in total will be ",runcounter)
# training ======================================================================================================

# build model
print("BUILD MODEL")
model = ResNet([3, 8, 36, 3])
model = model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    

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
while(training):
    
    FinalIDs_train = []
    # thisroundtrainfarm = sample(TrainFarm,len(TrainRiver))
    thisroundtrainfarm  = TrainFarm.copy()

    FinalIDs_train.extend(thisroundtrainfarm)
    FinalIDs_train.extend(TrainRiver)
    oversampleriver = sample(TrainRiver,(len(TrainFarm)-len(TrainRiver)))
    FinalIDs_train.extend(oversampleriver)
    trainother = [x for x in trainother if x not in thisroundtrainother]
    otherlen = len(TrainFarm)
    if(len(trainother)<otherlen):
        print("finished training!!!!!!!!!!!!!!!!!!!!!")
        df = pd.DataFrame(total_losslist)
        writer = pd.ExcelWriter(Model_Dir+'finished_test400.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name='welcome', index=False)
        writer.save()
        break
    thisroundtrainother = sample(trainother,otherlen)
    FinalIDs_train.extend(thisroundtrainother)
    


    frac_map_images_list_train = []
    label_images_list_train = []
    print(len(FinalIDs_train))
    print("begin to generate the images for train")
# Data loader===============================================================================================

    for ID in FinalIDs_train:
        frac_map_image, label_image, ID = create_frac_map_and_label(ID,train_label_load)            
        frac_map_images_list_train.append(frac_map_image)
        label_images_list_train.append(label_image)
    data_train = CLASSIFIER(frac_map_images_list_train, label_images_list_train,FinalIDs_train)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, num_workers=0)
# ===============================================================================================

    if(experiment_id!=0):
        model.load_state_dict(torch.load(os.path.join(Model_Dir, str(experiment_id)+".pt")))
        model = model.to('cuda')
    experiment_id +=1
    model.train()
    print(experiment_id)
    
    for epoch in range(1,no_epochs+1):
        if(num%batch_display == 0):
            print(str(num/controlepoch*100)+'%')
        num+=1
        total_losslist.append(total_loss)
        # if(epoch <= 60//runcounter):
        #     learning_rate = 0.1
        # elif epoch > 60//runcounter and epoch <= 120//runcounter:
        #     learning_rate = 0.02
        # elif epoch >120//runcounter and epoch <= 160//runcounter:
        #     learning_rate = 0.004
        # elif epoch >160//runcounter and epoch <= 200//runcounter:
        #     learning_rate = 0.0008
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
        for batch, [frac_map_batch, label_image_batch,ID_batch] in enumerate(data_loader_train):
            output = model(frac_map_batch.to('cuda').float())
            target_label = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch

            loss = criterion(output, target_label)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    torch.save(model.state_dict(), os.path.join(Model_Dir,str(experiment_id)+'.pt'))


# torch.save(model.state_dict(), os.path.join(Model_Dir,str(Model_Dir)+'_finished.pt'))
# testing ======================================================================================================

test_experiment_id = 5

model.load_state_dict(torch.load(os.path.join(Model_Dir, str(test_experiment_id)+".pt")))
model.eval()

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

IDS = FinalIDs_test.copy()
for ID in IDS:
    if(len(frac_map_images_list_test)%1000==0):
        print(len(frac_map_images_list_test))
    frac_map_image, label_image, ID = create_frac_map_and_label(ID,test_label_load)            
    frac_map_images_list_test.append(frac_map_image)
    label_images_list_test.append(label_image)
data_test = CLASSIFIER(frac_map_images_list_test, label_images_list_test,FinalIDs_test)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=0)
# ===============================================================================================

preds = []
labels = []
softmax_list =[]
IDs_all = []
for batch, [frac_map_batch, label_image_batch,ID_batch] in enumerate(data_loader_test):
    output = model(frac_map_batch.to('cuda').float()) # gets output for a batch
    target_label = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch

    softmax_batch = torch.nn.functional.softmax(output, dim=1)
    ID_batch_cpu = ID_batch.detach().cpu().numpy()
    out_label_batch = torch.argmax(softmax_batch, dim=1)
    out_label_batch_cpu = out_label_batch.detach().cpu().numpy()
    label_batch_cpu = target_label.detach().cpu().numpy()

    softmax_batch_cpu = softmax_batch.detach().cpu().numpy()
    for b in range(softmax_batch_cpu.shape[0]):
        softmax_list.append(softmax_batch_cpu[b])
        IDs_all.append(ID_batch_cpu[b])
        preds.append(out_label_batch_cpu[b])
        if(label_batch_cpu[b] == 3):
            labels.append(0)
        else:
            labels.append(label_batch_cpu[b])
    
pred_array = np.array(preds)
label_array = np.array(labels)

print(pred_array.shape)
print(label_array.shape)

print("np.bincount(pred_array) ",np.bincount(pred_array))
print("np.bincount(label_array))",np.bincount(label_array))
print(classification_report(label_array, pred_array, digits=4))
print(f1_score(y_true=label_array, y_pred=pred_array, average='macro'))