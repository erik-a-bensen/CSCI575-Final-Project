import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# read in the data from pixels.xlx sheet 1
data = pd.read_excel('../Data/pixels.xlsx', sheet_name=0)
# split data into a list of dataframes by Scene
scenes = [data[data['Scene'] == i] for i in data['Scene'].unique()]
classes = [{} for i in range(len(scenes))]
for s in range(len(scenes)):
    dict = classes[s]
    for c in scenes[s]['Pixel Class'].unique():
        dict[c] = scenes[s][scenes[s]['Pixel Class'] == c][['X-location', 'Y-location']].values

# read in the k-means data from ../Kmeans/kmeas.csv
kmeans = pd.read_csv('../Kmeans/kmeas.csv')

images = ['_rgb_hist_hmrf.png', '_SWIR_hist_hmrf.png', '_TIR_a_hist_hmrf.png']
n = 40
hmrf_scores = {}

#begin processing loop
for s in range(len(scenes)):
    scene_num = s+1
    hmrf_path = f'../HMRF Results/scene{scene_num}/'
    gt_pixels = classes[s]
    for image in images:
        path = hmrf_path + str(scene_num) + image
        # read in the image
        #try:
        img = plt.imread(path)
        print(img.shape)
        #convert to grayscale
        #img = img[:,:,0]
        #find unique pixel values
        unique_pixels = np.unique(img)
        print(f'Unique Pixels: {unique_pixels}')
        #assign integer values to each unique pixel in the image
        for i in range(len(unique_pixels)):
            img[img == unique_pixels[i]] = i

        correct = 0
        #loop through gt_pixels keys
        for key in gt_pixels.keys():
            values = []
            #loop through gt_pixels values
            for value in gt_pixels[key]:
                print(value)
                #find the pixel value in the image
                pixel_value = img[value[0]][value[1]]
                #add the pixel value to the list
                values.append(pixel_value)
            #find the mode of the list
            mode = np.bincount(values).argmax() 

            #calculate the correctly classified pixels for this key
            #loop through gt_pixels values
            for value in gt_pixels[key]:
                #find the pixel value in the image
                pixel_value = img[value[0]][value[1]]
                #check if the pixel value is equal to the mode
                if pixel_value == mode:
                    correct += 1
        #calculate the image score
        score = (2*correct - n)/n
        #add the score to the dictionary
        hmrf_scores[str(scene_num) + image] = score

        #except:
        #    pass #if the image doesn't exist, skip it

print(hmrf_scores)

        
