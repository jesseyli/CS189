import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import math

train = scipy.io.loadmat('data/mnist_data/images.mat')
train_images= train["images"]
train_images = np.float64(train_images.reshape(-1, train_images.shape[-1])).T
np.random.shuffle(train_images)

def montage_images(images):
    num_images=min(1000,np.size(images,2))
    numrows=math.floor(math.sqrt(num_images))
    numcols=math.ceil(num_images/numrows)
    img=np.zeros((numrows*28,numcols*28));
    for k in range(num_images):
        r = k % numrows
        c = k // numrows
        img[r*28:(r+1)*28,c*28:(c+1)*28]=images[:,:,k];
    return img


def kmeans(k):
	mu = [np.random.rand(train_images.shape[1])]*k
	classes = np.array([[] for _ in range(k)])

	iterate = True
	iteration = 0
	prev_loss = float('inf')
	while iterate:
		iteration += 1
		loss = 0
		for i, lst in enumerate(classes):
			for j in lst:
				loss += np.sum((j - mu[i])**2)
		if prev_loss == loss:
			iterate = False
		prev_loss = loss
		new_classes = [[] for _ in range(k)]
		for image in train_images:
			best_i, best_mu, best = 0, mu[0], float('inf')
			for i in range(k):
				cost = np.sum((mu[i] - image)**2)
				if cost < best:
					best_i, best_mu, best  = i, mu[i], cost
			new_classes[best_i].append(image)
		if not np.array_equal(classes,np.array(new_classes)):
			classes = np.array(new_classes)
			for i in range(k):
				if classes[i]:
					mu[i] = np.sum(np.array(classes[i]),axis=0)/len(classes[i])
	print(iteration)
	print(loss)
	return mu

k_list = [5,10,20]
for i,k in enumerate(k_list):
	mu = kmeans(k)
	for mean in mu:
		img = montage_images(mean.reshape(28,28,1))
		plt.imshow(img)
		plt.show()
