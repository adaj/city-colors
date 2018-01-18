import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2

def load_image(path, show=False):
	# load the image and convert it from BGR to RGB so that
	# we can dispaly it with matplotlib
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	if show:
		plt.figure()
		plt.axis("off")
		plt.imshow(image);

	# reshape the image to be a list of pixels
	image = image.reshape((image.shape[0] * image.shape[1], 3))
	return image

def image_kcolors(k, image):
	km = KMeans(n_clusters = k)
	km.fit(image)
	centroids = km.cluster_centers_
	return km, centroids

def kcolors_pct(km, centroids, show=False):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(km.labels_)) + 1)
	(hist, _) = np.histogram(km.labels_, bins = numLabels)
	hist = hist.astype("float")
	hist /= hist.sum()

	if show:
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

	return hist
