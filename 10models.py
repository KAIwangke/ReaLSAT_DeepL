# %%
from ast import Global
import os
import this
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from torch.utils.data.dataset import Dataset
import time
import glob
import torch

learning_rate = 0.001
patch_size = 64
inchannels = 1
outchannels = 2 
no_epochs = 370
lower_lim = 350
upper_lim = 400
batch_size = 8
continent_no = 2

# paths setting

farm_label_array_path = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/farm_labels.npy'
warped_data_path = '../../WARPED_DATA/350_400_stage2_warped_64x64/'
continent_info_array_path = '/home/kumarv/pravirat/Realsat_labelling/continent_info.npy'
paths_list = glob.glob(os.path.join(warped_data_path + '*.npy'))


# models and functions setting
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


# get the continent id info
continent_info = np.load(continent_info_array_path)
continent_path_list = []
continent_ID_list = []
for path in paths_list:
    ID = path.split('/')[-1].split('_')[-4]
    if(continent_info[int(ID)] == continent_no):
        continent_path_list.append(path)
        continent_ID_list.append(int(ID))
        # contain all the IDs in the list


criterion = torch.nn.CrossEntropyLoss()
thisid = 0
model = CNN_simple(in_channels=inchannels, out_channels=outchannels)
farm_label_array = np.load(farm_label_array_path)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

totallist= farm_label_array.tolist()

for i in range(1000000):
    totallist[i] = 0


IDs_all_copy = []
final_IDs = continent_ID_list.copy()
frac_map_images_list = []
label_images_list = []

# create
# count_array = np.zeros((10000))


print("begin to generate the images")
for ID in final_IDs:
    frac_map_image, label_image, ID = create_frac_map_and_label(ID,farm_label_array)            
    frac_map_images_list.append(frac_map_image)
    print(len(frac_map_images_list))
    label_images_list.append(label_image)
    # print(len(label_images_list))

print("start to run (image prepaired)")
# print the


Nofalgorithms = 10
round  = 1


def modelsrun():
    thisid = 0
    for i in range(Nofalgorithms):
        thisid+=1
        Model_Dir = '../models/test/'
        model_name = str(thisid)+'.pt'
        print(model_name)
        global model
        model.load_state_dict(torch.load(os.path.join(Model_Dir, model_name)))
        model = model.cuda()

        train_loss = []
        image_patches_list = []
        label_patches_list = []

        # print(len(final_IDs))
        data = CLASSIFIER(frac_map_images_list, label_images_list,final_IDs)
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=2, shuffle=False, num_workers=0)
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
            out_label_batch = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
            # print("this is the round"+str(i))
            # print(ID_batch,out,out_label_batch)
            loss += batch_loss.item()
            preds.append(out_label_batch.detach().cpu().numpy())
            labels.append(label_batch.cpu().numpy())
        #    same order
            IDs_all.append(ID_batch)

            # print()

        loss = loss/(batch+1)

        pred_array = np.concatenate(preds, axis=0)
        # label_array = np.concatenate(labels, axis=0)
        ID_array = np.concatenate(IDs_all, axis=0)
        global IDs_all_copy 
        IDs_all_copy = ID_array.copy()
        # print(len(pred_array))
        # print("ID_array"+str(len(ID_array)))
        for i,d in enumerate(ID_array):
            totallist[d] += pred_array[i]   

    return totallist

print(len(totallist))

if(max(totallist)>round*Nofalgorithms):
    exit()


new = []
notnew = []


for i in range(round):
    print("this is the round "+str(i))
    modelsrun()
    

counting10 = 0
counting0 = 0
counting1 = 0
counting2 = 0
counting3 = 0
counting4 = 0
counting5 = 0
counting6 = 0
counting7 = 0
counting8 = 0
counting9 = 0




for i,d in enumerate(totallist):
    # if d == 0:
    #     counting0+=1
    #     notnew.append(i)
    # if d>=1 and d<=(round*10/6):
    #     counting1+=1
    # if d>=(round*10/6+1) and d<=(round*10/3):
    #     counting2+=1
    # if d>=(round*10/3+1) and d<=(round*10/2):
    #     counting3+=1
    # if d>=(round*10/2+1) and d<(round*10*0.8):
    #     counting4+=1
    # if d>=(round*10*0.8)  and d<(round*10*0.9):
    #     counting5+=1
    # if d>(round*10*0.9):    
    #     new.append(i)
    #     counting6+=1
    # if(i in int(final_IDs)):
    #     if(d == 0):
    #         counting0+=1
    if(d == 1):
        counting1+=1
    if(d == 2):
        counting2+=1
    if(d == 3):
        counting3+=1
    if(d == 4):
        counting4+=1
    if(d == 5):
        counting5+=1
    if(d == 6):
        counting6+=1
    if(d == 7):
        counting7+=1
    if(d == 8):
        counting8+=1
    if(d == 9):
        counting9+=1
    if(d == 10):
        counting10+=1
        new.append(i)


print(len(final_IDs))


items = new.copy()
notitems = notnew.copy()
# open file in write mode

print("\n")

# print("range is 1 to "+str((round*10/6))+": "+str(counting1))
# print("range is "+str((round*10/6)+1)+"to "+str((round*10/3))+": "+str(counting2))
# print("range is "+str((round*10/3)+1)+"to "+str((round*10/2))+": "+str(counting3))
# print("=========")
# print("range is "+str((round*10/2)+1)+"to "+str(80)+": "+str(counting4))
# print("range is "+str((round*10*0.8))+"to "+str((round*10*0.9))+": "+str(counting5))
# print("range is over 90(what we want)"+": "+str(counting6))
# print("=========")
# print("perdiction as not farm is "+str(len(notnew)))
# print("new Appended is "+str(len(new)))

print(counting0)
print(counting1)
print(counting2)
print(counting3)
print(counting4)
print(counting5)
print(counting6)
print(counting7)
print(counting8)
print(counting9)
print(counting10)

with open(r'/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH/predictAsFarm.txt', 'w') as fp:
    for item in items:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Store Done')

with open(r'/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH/predictNotasFarm.txt', 'w') as fp:
    for item in notitems:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Store Done')
# print(len(totallist))
# print(totallist)
# print("this is the final output")
# print(new)