import enum
import numpy as np
from random import sample

US_path = '/home/kumarv/pravirat/Realsat_labelling/UnifiedModel/new_datasets/ARAF_test copy.npy'

loadnp = np.load(US_path)
print(np.bincount(loadnp))    

tolist= loadnp.tolist()


for i,d in enumerate(tolist):
    if d == 3:
        # print("1")
        loadnp[i]=0

np.save(US_path,loadnp)
print(np.bincount(loadnp))    
