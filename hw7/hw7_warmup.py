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


# Problem 2.2, basic classifier
# -----------------------------------------------------------------------
# def basic_recommend(jokes):
# 	predictions = []
# 	for joke in jokes:
# 		if joke_ratings[joke] > 0:
# 			predictions.append(1)
# 		else:
# 			predictions.append(0)
# 	return np.array(predictions)

def accuracy(vec1,vec2):
	return sum(vec1 == vec2)/vec1.shape[0]

# predict = basic_recommend(valid_pairs[:,1])
# basic_accuracy = accuracy(predict,validation[:,2]) # 0.62032520325203255
# ------------------------------------------------------------------------

# Problem 2.2, k-NN classifier
# ------------------------------------------------------------------------
# neighbors = np.zeros((100, r.shape[0]))
# for user in range(100):
# 	distances = []
# 	for i in range(r.shape[0]):
# 		distances.append((i,np.linalg.norm(r[i] - r[user])))
# 	neighbors[user] = np.array(sorted(distances, key=itemgetter(1)))[:,0]

# k_list = [10,100,1000]
# KNN_accuracies = []

# for k in k_list:
# 	predictions = []
# 	for point in validation:
# 		avg = 0
# 		for neighbor in neighbors[point[0]][1:k+1]:
# 			avg += r[neighbor][point[1]]
# 		if avg > 0: # only care if average of neighbors is positive
# 			predictions.append(1)
# 		else:
# 			predictions.append(0)
# 	predict = np.array(predictions)
# 	acc = accuracy(predict,validation[:,2])
# 	print(acc)
# 	KNN_accuracies.append(acc)
# ------------------------------------------------------------------------


# Problem 2.3.2, SVD classifier
# ------------------------------------------------------------------------

# def MSE(U,D,V):
# 	error = 0
# 	D = np.diag(D)
# 	r_prime = U.dot(D.dot(V))
# 	for i in range(r.shape[0]):
# 		for j in range(r.shape[1]):
# 			if not np.isnan(train['train'][i][j]):
# 				error += (r_prime[i][j] - r[i][j])**2
# 	return error

# def SVD_predict(U,D,V):
# 	D = np.diag(D)
# 	r_prime = U.dot(D.dot(V))
# 	predictions = []
# 	for point in validation:
# 		if r_prime[point[0]][point[1]] > 0:
# 			predictions.append(1)
# 		else:
# 			predictions.append(0)
# 	return np.array(predictions)

# U, D, V = np.linalg.svd(r, full_matrices=False)
# d_list = [2,5,10,20]
# MSE_list = []
# SVD_accuracies = []
# for d in d_list:
# 	MSE_list.append(MSE(U[:,:d],D[:d],V[:d]))
# 	predict = SVD_predict(U[:,:d],D[:d],V[:d])
# 	SVD_accuracies.append(accuracy(predict,validation[:,2]))
# plt.figure()
# plt.plot(d_list, MSE_list)
# plt.figure()
# plt.plot(d_list, SVD_accuracies)
# plt.show()
# ------------------------------------------------------------------------


class latent_factor_model:
    def __init__(self, lab):
        self.lab = lab
        self.u = np.random.random(U.shape)
        self.v = np.random.random(V.shape)
    
    def minimize_u(self):
        v_outer = []
        for l in range(len(self.v)):
            v_outer.append(np.outer(self.v[l], self.v[l]))
        
        for i in range(len(self.u)):
            inv = sum([v_outer[l] for l in range(len(self.v)) if S[i,l]])
            inv += self.lab*np.identity(len(inv))
            other = sum([self.v[l]*j_up_train[i,l] for l in range(len(self.v)) if S[i,l]])
            self.u[i] =  np.linalg.inv(inv).dot(other)
        pass
    
    def minimize_v(self):
        u_outer = []
        for k in range(len(self.u)):
            u_outer.append(np.outer(self.u[k], self.u[k]))
        
        for j in range(len(self.v)):
            inv = sum([u_outer[k] for k in range(len(self.u)) if S[k,j]])
            inv += self.lab*np.identity(len(inv))
            other = sum([self.u[k]*j_up_train[k,j] for k in range(len(self.u)) if S[k,j]])
            self.v[j] =  np.linalg.inv(inv).dot(other)
        pass
    
    
    def trainer(self):
        while True:
            self.minimize_u()
            print("Minimized u")
            self.minimize_v()
            print("Minimized v")


            yield self.lfm
        
    def lfm(self, user, joke):
        return self.u[user].dot(self.v[joke])


model = latent_factor_model(10)
trainer = model.trainer()
latent_accuracies = []
for i in range(10):
    print("Epoch", i)
    latent_accuracies.append(accuracy(next(trainer)))
    print(errors[i])