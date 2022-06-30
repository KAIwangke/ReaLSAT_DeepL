counting1 = 0
counting2 = 0
counting3 = 0
counting4 = 0

for i,d in enumerate(totallist):
    if d>=1 and d<=(round*10/6):
        counting1+=1
    if d>=(round*10/6+1) and d<=(round*10/3):
        counting2+=1
    if d>=(round*10/3+1) and d<=(round*10/2):
        counting3+=1
    if d>=(round*10/2+1) and d<=(round*10):
        counting4+=1

    if(totallist[i]>=13):
        new.append(IDs_all_copy[i])
        # print("current adding ID is : "+str(IDs_all_copy(i)))


items = new.copy()
# open file in write mode

print("\n")

print(counting1)
print(counting2)
print(counting3)

print("=========")
print(counting4)
