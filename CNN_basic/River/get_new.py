import numpy as np

test_result_10 = []
test_result_9 = []
test_result_8 = []
test_result_7 = []
test_result_6 = []
test_result_5 = []
test_result_4 = []
test_result_3 = []
test_result_2 = []
test_result_1 = []
test_result_0 = []

origin = []

f_10 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level10s.txt','r')
f_9 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level9s.txt','r')
f_8 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level8s.txt','r')
f_7 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level7s.txt','r')
f_6 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level6s.txt','r')
f_5 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level5s.txt','r')
f_4 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level4s.txt','r')
f_3 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level3s.txt','r')
f_2 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level2s.txt','r')
f_1 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level1s.txt','r')
f_0 = open('/home/kumarv/pravirat/Realsat_labelling/TEMPSTOREPATH1/level0s.txt','r')

river_label_array = np.load('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/river_labels.npy')
river_label_array = list(river_label_array)

for line in f_10:
    test_result_10.append(int(line.strip()))

for line in f_9:
    test_result_9.append(int(line.strip()))

for line in f_8:
    test_result_8.append(int(line.strip()))

for line in f_7:
    test_result_7.append(int(line.strip()))

for line in f_6:
    test_result_6.append(int(line.strip()))

for line in f_5:
    test_result_5.append(int(line.strip()))

for line in f_4:
    test_result_4.append(int(line.strip()))

for line in f_3:
    test_result_3.append(int(line.strip()))

for line in f_2:
    test_result_2.append(int(line.strip()))

for line in f_1:
    test_result_1.append(int(line.strip()))

for line in f_0:
    test_result_0.append(int(line.strip()))

for i in range(0,len(river_label_array)):
    if river_label_array[i]==1:
        origin.append(i)


for i in origin:
        if i in test_result_10:
            test_result_10.remove(i)
        
        elif i in test_result_9:
            test_result_9.remove(i)

        elif i in test_result_8:
            test_result_8.remove(i)

        elif i in test_result_7:
            test_result_7.remove(i)

        elif i in test_result_6:
            test_result_6.remove(i)

        elif i in test_result_5:
            test_result_5.remove(i)

        elif i in test_result_4:
            test_result_4.remove(i)

        elif i in test_result_3:
            test_result_3.remove(i)

        elif i in test_result_2:
            test_result_2.remove(i)

        elif i in test_result_1:
            test_result_1.remove(i)

        elif i in test_result_0:
            test_result_0.remove(i)

print("origin list:")
print(len(origin))

print("new count == 10:")
print(test_result_10)
print(len(test_result_10))

print("new count == 9:")
print(test_result_9)
print(len(test_result_9))

print("new count == 8:")
print(test_result_8)
print(len(test_result_8))

print("new count == 7:")
print(len(test_result_7))

print("new count == 6:")
print(len(test_result_6))

print("new count == 5:")
print(len(test_result_5))

print("new count == 4:")
print(len(test_result_4))

print("new count == 3:")
print(len(test_result_3))

print("new count == 2:")
print(len(test_result_2))

print("new count == 1:")
print(len(test_result_1))

print("new count == 0:")
print(len(test_result_0))
