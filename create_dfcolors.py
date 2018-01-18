"""
This code creates the dataframe with the colors to be labeled
"""

import os
import numpy as np
import pandas as pd
from script import image_kcolors, load_image

path = 'photos/'
k = 10

for i, photofile in enumerate(os.listdir(path)):
	if photofile.endswith('.png'):
		print('Processing {}'.format(path+photofile))
		kmeans, centroids = image_kcolors(k, load_image(path+photofile))
		if i==0:
			colors=np.concatenate((centroids,np.full((centroids.shape[0],1),photofile[0])),axis=1)
		else:
			colors=np.concatenate((colors,np.concatenate((centroids,np.full((centroids.shape[0],1),photofile[0])),axis=1)),axis=0)
print('Finish')

dfcolors = pd.DataFrame(colors)
dfcolors['label'] = np.zeros(len(df))
dfcolors
dfcolors.to_csv('colors_df.csv',index_label='i')
