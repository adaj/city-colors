%matplotlib inline

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2


args = {
	'image': 'photos/mato_zoom18.png',
	'clusters': 20
}

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(image);

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))
image.shape

# cluster the pixel intensities
km = KMeans(n_clusters = args['clusters'])
km.fit(image)
centroids = km.cluster_centers_

# grab the number of different clusters and create a histogram
# based on the number of pixels assigned to each cluster
numLabels = np.arange(0, len(np.unique(km.labels_)) + 1)
(hist, _) = np.histogram(km.labels_, bins = numLabels)

# normalize the histogram, such that it sums to one
hist = hist.astype("float")
hist /= hist.sum()

# initialize the bar chart representing the relative frequency
# of each of the colors
bar = np.zeros((50, 300, 3), dtype = "uint8")
startX = 0

# loop over the percentage of each cluster and the color of
# each cluster
for (percent, color) in zip(hist, centroids):
	# plot the relative percentage of each cluster
	endX = startX + (percent * 300)
	cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
		color.astype("uint8").tolist(), -1)
	startX = endX

# initialize the bar chart representing the relative frequency
# of each of the colors
bar = np.zeros((50, 300, 3), dtype = "uint8")
startX = 0

# loop over the percentage of each cluster and the color of
# each cluster
for (percent, color) in zip(hist, centroids):
	# plot the relative percentage of each cluster
	endX = startX + (percent * 300)
	cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
		color.astype("uint8").tolist(), -1)
	startX = endX

plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()


def centroid_weight(centroids, bar):
	centroids = centroids.astype(int)
	P = []
	c=0
	for c in centroids:
		count=0
		for a in bar[0]==c:
			if [True,True,True] in a:
				count+=1
		P.append(round(count/bar[0].shape[0],2))
	return P

print(centroids)
print()
print(centroid_weight(centroids, bar))


#
