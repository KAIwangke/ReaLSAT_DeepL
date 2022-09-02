from os.path import exists



ID = []
file_name = 'check6.txt'

determine_list = ID

# eph = []
farm = []
not_ef = []

i = 0

if exists('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/'+file_name) == False:
    f = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/'+file_name,'w')
    f.close()

else:
    f = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/'+file_name,'r')

    for line in f:
        i += 1
        
### If the program is killed and you rerun it, uncomment below part to retrieve the current progress

while i < len(determine_list):
    f = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/'+file_name,'a')
    d = input("determine " + str(determine_list[i]) + " is farm (1) or not(0)  ")
    if d == '1':
        f.write(str(0)*(6-len(str(determine_list[i]))) + str(determine_list[i]) + ' F ' + '\n')
        i = i + 1
        f.close()
    elif d == '0':
        f.write(str(0)*(6-len(str(determine_list[i]))) + str(determine_list[i]) + ' N ' + '\n')
        i = i + 1
        f.close()
    # elif d == '2':
    #     f.write(str(0)*(6-len(str(determine_list[i]))) + str(determine_list[i]) + ' E ' + '\n')
    #     i = i + 1
    #     f.close()
    else:
        print("Invalid Input, Please recheck")

f = open('/home/kumarv/pravirat/Realsat_labelling/river_code_kx/txt/'+file_name,'r')
for line in f:
    # if line.find('E') != -1:
    #     eph.append(int(line[:6]))
    if line.find('F') != -1:
        farm.append(int(line[:6]))


# print("This is ephermeral")
# print(eph)
# print(len(eph))

# print("##########################")

print("This is farm")
print(farm)
print(len(farm))

print("##########################")

for i in determine_list:
    # if i not in eph:
    if i not in farm:
        not_ef.append(i)

print("This is not farm")
print(not_ef)
print(len(not_ef))
