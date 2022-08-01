# %%
import numpy as np
# import 

datapath_test = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/FR_test.npy'
datapath_test_load = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/FR_test_load.npy'
datapath_train = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/FR_train.npy'
datapath_train_load = '/home/kumarv/pravirat/Realsat_labelling/farm_code_W/UnifiedModel/new_datasets/FR_train_load.npy'


datapath_test = np.load(datapath_test)
datapath_test = datapath_test.tolist()

datapath_test_load = np.load(datapath_test_load)
datapath_test_load = datapath_test_load.tolist()

datapath_train = np.load(datapath_train)
datapath_train = datapath_train.tolist()


datapath_train_load = np.load(datapath_train_load)
datapath_train_load = datapath_train_load.tolist()



countriver_test = []
countfarm_test = []
countother_test = []

for i,d in enumerate(datapath_test):
    if d ==2:
        countriver_test.append(i)
    if d == 1:
        countfarm_test.append(i)
    if d == 3:
        countother_test.append(i)

print(len(countriver_test))
print(len(countfarm_test))
print(len(countother_test))

countriver_test_load = []
countfarm_test_load = []
countother_test_load = []

for i,d in enumerate(datapath_test_load):
    if d ==2:
        countriver_test_load.append(i)
    if d == 1:
        countfarm_test_load.append(i)
    if d == 3:
        countother_test_load.append(i)

print(len(countriver_test_load))
print(len(countfarm_test_load))
print(len(countother_test_load))


countriver_train = []
countfarm_train = []
countother_train = []

for i,d in enumerate(datapath_train):
    if d ==2:
        countriver_train.append(i)
    if d == 1:
        countfarm_train.append(i)
    if d == 3:
        countother_train.append(i)

print(len(countriver_train))
print(len(countfarm_train))
print(len(countother_train))



countriver_train_load = []
countfarm_train_load = []
countother_train_load = []

for i,d in enumerate(datapath_train_load):
    if d ==2:
        countriver_train_load.append(i)
    if d == 1:
        countfarm_train_load.append(i)
    if d == 3:
        countother_train_load.append(i)

print(len(countriver_train_load))
print(len(countfarm_train_load))
print(len(countother_train_load))
# ==========================================================
if(countriver_train_load==countriver_train):
    print(True)







# %%
