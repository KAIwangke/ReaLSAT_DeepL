from ast import Not
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

Model_Dir = '/home/kumarv/pravirat/Realsat_labelling/UnifiedModel//Models_1/'

## Parameters
experiment_id = '10models_X'
learning_rate = 0.001
patch_size = 64
inchannels = 1
outchannels = 3 
# 0 represents unconfirmed
# 1 represents confirmed as farm
# 2 represents confirmed as not farm
no_epochs = 500
lower_lim = 350
upper_lim = 400
batch_size = 256
continent_no = 1
bias =  0.0000000175

test_label_array_path = '/home/kumarv/pravirat/Realsat_labelling/UnifiedModel/Test_store.npy'
load_test_label_array_path = '/home/kumarv/pravirat/Realsat_labelling/UnifiedModel/Test.npy'


warped_data_path = '/home/kumarv/pravirat/Realsat_labelling/WARPED_DATA/350_400_stage2_warped_64x64/'
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
model_name = '2.pt'



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
        self.fc = torch.nn.Linear(4096, 4096)

        self.upfc = torch.nn.Linear(4096, 4096)
        self.unpool3 = torch.nn.ConvTranspose2d(64 , 64, kernel_size=2, stride=2)
        self.upconv3_1 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.upconv3_2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.unpool2 = torch.nn.ConvTranspose2d(32 , 32, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(16 , 16, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(16, in_channels, 3, padding=1)
        
        self.classification_layer = torch.nn.Linear(4096, out_channels)

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

# build model

model = CNN_reconstruct(in_channels=inchannels, out_channels=outchannels)
model = model.to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


model.eval()





model.load_state_dict(torch.load(os.path.join(Model_Dir, model_name)))


farm_label_array = np.load(test_label_array_path)
testlist = farm_label_array.tolist()
print(np.bincount(farm_label_array))



# exit()
# paths_list = glob.glob(os.path.join(warped_data_path + '*.npy')) # gets all paths

# #subset paths that lie in continent of interest
# continent_info = np.load(continent_info_array_path)
# continent_path_list = []
# continent_ID_list = []
# for path in paths_list:
#     ID = path.split('/')[-1].split('_')[-4]
#     if(continent_info[int(ID)] == continent_no):
#         continent_path_list.append(path)
#         ## get all ID from certain continent
#         continent_ID_list.append(int(ID))

# print(len(continent_ID_list))

# farm_label_array = np.load(test_label_array_path)
# totallist1= farm_label_array.tolist()
# for i in range(1000000):
#     totallist1[i] = 0
# totallist2 = totallist1.copy()

# final_IDs = continent_ID_list.copy()
# print(len(final_IDs))
# frac_map_images_list = []
# label_images_list = []
# print("begin to generate the images")
# for ID in final_IDs:
#     if(len(label_images_list)%1000 == 0):
#         print(len(label_images_list))
#     frac_map_image, label_image, ID = create_frac_map_and_label(ID,farm_label_array)            
#     frac_map_images_list.append(frac_map_image)
#     label_images_list.append(label_image)

# print("start to run (image prepaired)")
# model.eval()
# data = CLASSIFIER(frac_map_images_list, label_images_list,final_IDs)
# data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=512, shuffle=False, num_workers=0)
# preds = []
# labels = []
# softmax_list =[]
# IDs_all = []
# for batch, [frac_map_batch, label_image_batch,ID_batch] in enumerate(data_loader):
#     classification,out = model(frac_map_batch.to('cuda').float()) # gets output for a batch
#     label_batch = label_image_batch.type(torch.long).to('cuda') # gets the labels for that batch
#     softmax_batch = torch.nn.functional.softmax(classification, dim=1)
#     ID_batch_cpu = ID_batch.detach().cpu().numpy()
#     # print(softmax_batch.shape)
#     # out_label_batch = torch.argmax(softmax_batch, dim=1)
#     # preds  = list(preds)
#     # preds.append(out_label_batch.detach().cpu().numpy())
#     # labels.append(label_batch.cpu().numpy())
#     # IDs_all.append(ID_batch)

#     softmax_batch_cpu = softmax_batch.detach().cpu().numpy()
#     for b in range(softmax_batch_cpu.shape[0]):
#         softmax_list.append(softmax_batch_cpu[b])
#         IDs_all.append(ID_batch_cpu[b])

# print(len(softmax_list))
# print(softmax_list[1])
# print(softmax_list[2])
# print(softmax_list[3])

# np.save('/home/kumarv/pravirat/Realsat_labelling/ASIA_farm/models/3channel_fullconnnect/softmax_array_22.npy',np.array(softmax_list))
# np.save('/home/kumarv/pravirat/Realsat_labelling/ASIA_farm/models/3channel_fullconnnect/ID_array_22.npy',np.array(IDs_all))

# farm_PATH = '/home/kumarv/pravirat/Realsat_labelling/ASIA_farm/farm_labels_asia.npy'
# import numpy as np

# farm_label_array = np.load(farm_PATH)

# farms = []
# rivers = []
# for i,d in enumerate(farm_label_array):
#     if d == 1:
#         farms.append(i)

       
# softmax_scores = np.load('/home/kumarv/pravirat/Realsat_labelling/ASIA_farm/models/3channel_fullconnnect/softmax_array_18.npy')
# print(softmax_scores.shape)

# other_score = softmax_scores[:,0]
# farm_score = softmax_scores[:,1]
# river_score = softmax_scores[:,2]

# for i,d in enumerate(farm_score):
#     if d == river_score[i]:
#         print(continent_ID_list[i])
#         print(farm_score[i])

# print(other_score.shape)
# print(farm_score.shape)
# print(river_score.shape)
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

# plt.savefig('/home/kumarv/pravirat/Realsat_labelling/ASIA_farm/models/3channel_fullconnnect/softmax_array_22.png')

# plt.show()















































