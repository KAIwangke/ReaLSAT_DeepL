import numpy as np
## 1. can automatically adjust ratio of train and test set

############ ONLY PARAMETER NEED TO MANUALLY ADJUST

train_ratio = 0.8

############


af_test_load_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/ARAF_test_load.npy'
af_test_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/ARAF_test.npy'
af_train_load_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/ARAF_train_load.npy'
af_train_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/ARAF_train.npy'

us_test_load_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/URUF_test_load.npy'
us_test_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/URUF_test.npy'
us_train_load_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/URUF_train_load.npy'
us_train_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/URUF_train.npy'

uni_test_load_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/FR_test_load.npy'
uni_test_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/FR_test.npy'
uni_train_load_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/FR_train_load.npy'
uni_train_path = '/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/FR_train.npy'

af_test_load = np.load(af_test_load_path)
af_train_load = np.load(af_train_load_path)

af_test = np.load(af_test_path)
af_train = np.load(af_train_path)

load_way =  len(np.bincount(af_train_load))  ## check how many ways
way = len(np.bincount(af_train))


#####################################
## combine af

af_total_load = (np.zeros(1000000)).astype(int)


for i in range(1000000):
    if af_test_load[i] != 0:
        af_total_load[i] = af_test_load[i]
    if af_train_load[i] != 0:
        af_total_load[i] = af_train_load[i]


af_total = (np.zeros(1000000)).astype(int)

for i in range(1000000):
    if af_test[i] != 0:
        af_total[i] = af_test[i]
    if af_train[i] != 0:
        af_total[i] = af_train[i]



#####################################
## re-distribute

count_load = list((np.zeros(load_way-1)).astype(int))
count = list((np.zeros(way-1)).astype(int))

for i in af_total_load:
    for j in range(1, load_way):
        if i == j:
            count_load[j-1] += 1 

for i in range(load_way-1):
    count_load[i] = int(count_load[i]*train_ratio)


for i in af_total:
    for j in range(1, way):
        if i == j:
            count[j-1] += 1 

for i in range(way-1):
    count[i] = int(count[i]*train_ratio)

#########################################
new_af_train_load = (np.zeros(1000000)).astype(int)
new_af_test_load = (np.zeros(1000000)).astype(int)

new_af_train = (np.zeros(1000000)).astype(int)
new_af_test = (np.zeros(1000000)).astype(int)

## new load train set
for i in range(1000000):
    for j in range(1, load_way):
        if af_total_load[i] == j and count_load[j-1] != 0:
            new_af_train_load[i] = af_total_load[i]
            count_load[j-1] = count_load[j-1]-1
            af_total_load[i] = af_total_load[i] * (-1)


## new load test set
for i in range(1000000):
    for j in range(1, load_way):
        if af_total_load[i] == j:
            new_af_test_load[i] = af_total_load[i]


## new train set
for i in range(1000000):
    for j in range(1, way):
        if af_total[i] == j and count[j-1] != 0:
            new_af_train[i] = af_total[i]
            count[j-1] = count[j-1]-1
            af_total[i] = af_total[i] * (-1)



## new test set
for i in range(1000000):
    for j in range(1, way):
        if af_total[i] == j:
            new_af_test[i] = af_total[i]

print("ORIGINAL ASIA TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(af_train))
print("TEST:")
print(np.bincount(af_test))

print("NEW ASIA TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(new_af_train))
print("TEST:")
print(np.bincount(new_af_test))

print("ORIGINAL ASIA LOAD TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(af_train_load))
print("TEST:")
print(np.bincount(af_test_load))

print("NEW ASIA LOAD TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(new_af_train_load))
print("TEST:")
print(np.bincount(new_af_test_load))

print('######################################')

#####################################
## combine us

us_test_load = np.load(us_test_load_path)
us_train_load = np.load(us_train_load_path)

us_test = np.load(us_test_path)
us_train = np.load(us_train_path)

load_way =  len(np.bincount(us_train_load))  ## check how many ways
way = len(np.bincount(us_train))

us_total_load = (np.zeros(1000000)).astype(int)


for i in range(1000000):
    if us_test_load[i] != 0:
        us_total_load[i] = us_test_load[i]
    if us_train_load[i] != 0:
        us_total_load[i] = us_train_load[i]


us_total = (np.zeros(1000000)).astype(int)

for i in range(1000000):
    if us_test[i] != 0:
        us_total[i] = us_test[i]
    if us_train[i] != 0:
        us_total[i] = us_train[i]



#####################################
## re-distribute
count_load = list((np.zeros(load_way-1)).astype(int))
count = list((np.zeros(way-1)).astype(int))

for i in us_total_load:
    for j in range(1, load_way):
        if i == j:
            count_load[j-1] += 1 

for i in range(load_way-1):
    count_load[i] = int(count_load[i]*train_ratio)


for i in us_total:
    for j in range(1, way):
        if i == j:
            count[j-1] += 1 

for i in range(way-1):
    count[i] = int(count[i]*train_ratio)

#########################################
new_us_train_load = (np.zeros(1000000)).astype(int)
new_us_test_load = (np.zeros(1000000)).astype(int)

new_us_train = (np.zeros(1000000)).astype(int)
new_us_test = (np.zeros(1000000)).astype(int)

## new load train set
for i in range(1000000):
    for j in range(1, load_way):
        if us_total_load[i] == j and count_load[j-1] != 0:
            new_us_train_load[i] = us_total_load[i]
            count_load[j-1] = count_load[j-1]-1
            us_total_load[i] = us_total_load[i] * (-1)


## new load test set
for i in range(1000000):
    for j in range(1, load_way):
        if us_total_load[i] == j:
            new_us_test_load[i] = us_total_load[i]


## new train set
for i in range(1000000):
    for j in range(1, way):
        if us_total[i] == j and count[j-1] != 0:
            new_us_train[i] = us_total[i]
            count[j-1] = count[j-1]-1
            us_total[i] = us_total[i] * (-1)



## new test set
for i in range(1000000):
    for j in range(1, way):
        if us_total[i] == j:
            new_us_test[i] = us_total[i]

print("ORIGINAL US TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(us_train))
print("TEST:")
print(np.bincount(us_test))

print("NEW US TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(new_us_train))
print("TEST:")
print(np.bincount(new_us_test))

print("ORIGINAL US LOAD TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(us_train_load))
print("TEST:")
print(np.bincount(us_test_load))

print("NEW US LOAD TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(new_us_train_load))
print("TEST:")
print(np.bincount(new_us_test_load))

########################################

## uni 

uni_test_load = np.load(uni_test_load_path)
uni_train_load = np.load(uni_train_load_path)

uni_test = np.load(uni_test_path)
uni_train = np.load(uni_train_path)

new_uni_train = (np.zeros(1000000)).astype(int)
new_uni_test = (np.zeros(1000000)).astype(int)

new_uni_train_load = (np.zeros(1000000)).astype(int)
new_uni_test_load = (np.zeros(1000000)).astype(int)

for i in range(1000000):
    if new_af_train[i] != 0:
        new_uni_train[i] = new_af_train[i]
    elif new_us_train[i] != 0:
        new_uni_train[i] = new_us_train[i]

    if new_af_test[i] != 0:
        new_uni_test[i] = new_af_test[i]
    elif new_us_test[i] != 0:
        new_uni_test[i] = new_us_test[i]

    if new_af_train_load[i] != 0:
        new_uni_train_load[i] = new_af_train_load[i]
    elif new_us_train_load[i] != 0:
        new_uni_train_load[i] = new_us_train_load[i]

    if new_af_test_load[i] != 0:
        new_uni_test_load[i] = new_af_test_load[i]
    elif new_us_test_load[i] != 0:
        new_uni_test_load[i] = new_us_test_load[i]

print('######################################')

print("ORIGINAL UNIFIED TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(uni_train))
print("TEST:")
print(np.bincount(uni_test))

print("NEW UNIFIED TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(new_uni_train))
print("TEST:")
print(np.bincount(new_uni_test))

print("ORIGINAL UNIFIED LOAD TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(uni_train_load))
print("TEST:")
print(np.bincount(uni_test_load))

print("NEW UNIFIED LOAD TRAIN AND TEST")
print("TRAIN:")
print(np.bincount(new_uni_train_load))
print("TEST:")
print(np.bincount(new_uni_test_load))

    
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/ARAF_test_load.npy', new_af_test_load)
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/ARAF_train_load.npy', new_af_train_load)
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/ARAF_test.npy', new_af_test)
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/ARAF_train.npy', new_af_train)

# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/URUF_test_load.npy', new_us_test_load)
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/URUF_train_load.npy', new_us_train_load)
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/URUF_test.npy', new_us_test)
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/URUF_train.npy', new_us_train)

# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/FR_test_load.npy', new_uni_test_load)
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/FR_train_load.npy', new_uni_train_load)
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/FR_test.npy', new_uni_test)
# np.save('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/SET/350_400/FR_train.npy', new_uni_train)
