import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import math
import pdb
from operator import itemgetter

train = scipy.io.loadmat('data/joke_data/joke_train.mat')
r = np.nan_to_num(train['train'])
joke_ratings = np.sum(r,axis=0) # only care if average is positive for basic classifier

validation = np.genfromtxt('./data/joke_data/validation.txt', delimiter=',') - np.array([1,1,0])
valid_pairs = validation[:,:2]

def basic_recommend(jokes):
	predictions = []
	for joke in jokes:
		if joke_ratings[joke] > 0:
			predictions.append(1)
		else:
			predictions.append(0)
	return np.array(predictions)

def accuracy(vec1,vec2):
	return sum(vec1 == vec2)/vec1.shape[0]

predict = basic_recommend(valid_pairs[:,1])
basic_accuracy = accuracy(predict,validation[:,2]) # 0.62032520325203255



neighbors = np.zeros((100, r.shape[0]))
for user in range(100):
	distances = []
	for i in range(r.shape[0]):
		distances.append((i,np.linalg.norm(r[i] - r[user])))
	neighbors[user] = np.array(sorted(distances, key=itemgetter(1)))[:,0]

k_list = [10,100,1000]
KNN_accuracies = []

for k in k_list:
	predictions = []
	for point in validation:
		avg = 0
		for neighbor in neighbors[point[0]][1:k+1]:
			avg += r[neighbor][point[1]]
		if avg > 0: # only care if average of neighbors is positive
			predictions.append(1)
		else:
			predictions.append(0)
	predict = np.array(predictions)
	acc = accuracy(predict,validation[:,2])
	print(acc)
	KNN_accuracies.append(acc)









