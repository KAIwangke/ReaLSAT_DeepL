from os.path import exists

ID = []

determine_list = ID

eph = []
not_eph = []

i = 0

if exists('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/eph.txt') == False:
    f = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/eph.txt','w')
    f.close()

else:
    f = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/eph.txt','r')

    for line in f:
        i += 1
        


while i < len(determine_list):
    f = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/eph.txt','a')
    d = input("determine " + str(determine_list[i]) + " is ephermeral (1) or not (0)  ")
    if d == '1':
        f.write(str(0)*(6-len(str(determine_list[i]))) + str(determine_list[i]) + ' Y ' + '\n')
        i = i + 1
        f.close()
    elif d == '0':
        f.write(str(0)*(6-len(str(determine_list[i]))) + str(determine_list[i]) + ' N ' + '\n')
        i = i + 1
        f.close()
    else:
        print("Invalid Input, Please recheck")

f = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/eph.txt','r')
for line in f:
    if line.find('Y') != -1:
        eph.append(int(line[:6]))

print("This is ephermeral")
print(eph)
print(len(eph))

print("##########################")

for i in determine_list:
    if i not in eph:
        not_eph.append(i)

print("This is not ephermeral")
print(not_eph)
print(len(not_eph))
