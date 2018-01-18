"""
This codes is used to plot all the colors fitted in dfcolors.
With this plot, you can add a column 'label' to dfcolors and
put 0 or 1 in order to classify as 0-'not green' and 1-'green'
"""


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

dfcolors = pd.read_csv('colors_df.csv',index_col='i')
colors = dfcolors[['0','1','2']].values

i=0
for i in range(0,len(dfcolors)):
	# Create a black image
	img = np.zeros((512,512,3), np.uint8)
	# Draw a red closed circle
	cv2.circle(img,(250,250), 200, colors[i], -1)
	#Display the image
	print(i,colors[i])
	plt.imshow(img)
	plt.show()
