# load ID of image of interest
ID = 626849
labeled_farm=[731630,731328,730504,730545,734435,734214,730304,68165,689616,731761,683693,730997,731571,731631,731813,731723]
labeled_none_farm_stage1 = [418841,463564,520007,547329,540241,622357,693164,558950,573960,554503,463607,414370,519837,5215855,40108,629826]

# i = 0
for i,d in enumerate(labeled_farm):
    ID = labeled_farm[i]
    for path in paths_list:
    #     print(path)
        ID_path = path.split('/')[-1].split('_')[-4]
        if int(ID) == int(ID_path):      
            path_final = path
            # print(ID)
            # print(i)
            image = np.load(path_final)
            print(path_final)
            # converting image to binary
            image[image == 1] = 0
            image[image == 2] = 1

            # take mean over timestamps to create fraction map
            frac_map = np.mean(image,axis = 0)
            # plotting the fraction map
            fig = plt.figure(figsize=(5,5))
            plt.title(labeled_farm[i])
            plt.imshow(frac_map[0])
            plt.show()
            #count number of water pixels timestamp wise
            water_pixels = []
            for t in range(image.shape[0]):
                no_water = np.sum(image[t] == 1)
                water_pixels.append(no_water)

            # plotting that time series 
            fig = plt.figure(figsize=(5,5))
            plt.title('Water count over time')
            plt.plot(water_pixels)
            plt.show()

# # create timeseries for ID

# # load ID numpy arary
# image = np.load(path_final)

# # converting image to binary
# image[image == 1] = 0
# image[image == 2] = 1

