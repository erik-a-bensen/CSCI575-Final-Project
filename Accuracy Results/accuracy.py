import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
print(kmeans)
        
