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
#   predictions = []
#   for joke in jokes:
#       if joke_ratings[joke] > 0:
#           predictions.append(1)
#       else:
#           predictions.append(0)
#   return np.array(predictions)

def accuracy(vec1,vec2):
    return sum(vec1 == vec2)/vec1.shape[0]

# predict = basic_recommend(valid_pairs[:,1])
# basic_accuracy = accuracy(predict,validation[:,2]) # 0.62032520325203255
# ------------------------------------------------------------------------

# Problem 2.2, k-NN classifier
# ------------------------------------------------------------------------
# neighbors = np.zeros((100, r.shape[0]))
# for user in range(100):
#   distances = []
#   for i in range(r.shape[0]):
#       distances.append((i,np.linalg.norm(r[i] - r[user])))
#   neighbors[user] = np.array(sorted(distances, key=itemgetter(1)))[:,0]

# k_list = [10,100,1000]
# KNN_accuracies = []

# for k in k_list:
#   predictions = []
#   for point in validation:
#       avg = 0
#       for neighbor in neighbors[point[0]][1:k+1]:
#           avg += r[neighbor][point[1]]
#       if avg > 0: # only care if average of neighbors is positive
#           predictions.append(1)
#       else:
#           predictions.append(0)
#   predict = np.array(predictions)
#   acc = accuracy(predict,validation[:,2])
#   print(acc)
#   KNN_accuracies.append(acc)
# ------------------------------------------------------------------------


# Problem 2.3.2, SVD classifier
# ------------------------------------------------------------------------

# def MSE(U,D,V):
#   error = 0
#   D = np.diag(D)
#   r_prime = U.dot(D.dot(V))
#   for i in range(r.shape[0]):
#       for j in range(r.shape[1]):
#           if not np.isnan(train['train'][i][j]):
#               error += (r_prime[i][j] - r[i][j])**2
#   return error

# def SVD_predict(U,D,V):
#   D = np.diag(D)
#   r_prime = U.dot(D.dot(V))
#   predictions = []
#   for point in validation:
#       if r_prime[point[0]][point[1]] > 0:
#           predictions.append(1)
#       else:
#           predictions.append(0)
#   return np.array(predictions)

U, D, V = np.linalg.svd(r, full_matrices=False)
# d_list = [2,5,10,20] # 9
# MSE_list = []
# SVD_accuracies = []
# for d in d_list:
#   MSE_list.append(MSE(U[:,:d],D[:d],V[:d]))
#   print(MSE(U[:,:d],D[:d],V[:d]))
#   predict = SVD_predict(U[:,:d],D[:d],V[:d])
#   print(accuracy(predict,validation[:,2]))
#   SVD_accuracies.append(accuracy(predict,validation[:,2]))
# plt.figure()
# plt.plot(d_list, MSE_list)
# plt.figure()
# plt.plot(d_list, SVD_accuracies)
# plt.show()
# ------------------------------------------------------------------------



# D = np.diag(D)
class latent_factor_model:
    def __init__(self, U, V, lamb):
        self.u = np.random.random(U.shape)
        self.v = np.random.random(V.shape)
        self.lamb = lamb
    
    def minimize(self, a, b):
        outer = []
        for i in range(b.shape[0]):
            outer.append(np.outer(b[i],b[i]))
        for i in range(a.shape[0]):
            try:
                first_half = sum([outer[j] for j in range(len(outer)) if not np.isnan(train['train'][i][j])])
                first_half += self.lamb*np.identity(len(outer))
                second_half = sum([b[j]*train['train'][i][j] for j in range(len(outer)) if not np.isnan(train['train'][i][j])])
            except IndexError:
                pdb.set_trace()
            # other = sum([self.v[l]*train['train'][i,l] for l in range(len(self.v)) if D[i,l]])
            # pdb.set_trace()
            a[i] =  np.linalg.inv(first_half).dot(second_half)
    
    def minimize_v(self):
        outer = []
        for k in range(len(self.u)):
            outer.append(np.outer(self.u[k], self.u[k]))
        for j in range(len(self.v)):
            inv = sum([outer[k] for k in range(len(self.u)) if D[k,j]])
            inv += self.lamb*np.identity(len(inv))
            other = sum([self.u[k]*train['train'][k,j] for k in range(len(self.u)) if D[k,j]])
            self.v[j] =  np.linalg.inv(inv).dot(other)
        pass
    
    
    def rating_generator(self):
        while True:
            self.minimize(self.u,self.v)
            self.minimize(self.v,self.u)
            yield self.rate
        
    def rate(self, user, joke):
        return self.u[user].dot(self.v[joke])


lamb_list = [0.01,0.1,1,10,100]
model = latent_factor_model(U, V, 10)
gen_rate = model.rating_generator()
latent_accuracies = []
for i in range(10):
    rate = next(gen_rate)
    predictions = []
    for sample in validation:
        predictions.append(rate(sample[0],sample[1]) > 0)
    acc = accuracy(np.array(predictions).astype(int), validation[:,2])
    latent_accuracies.append(acc)
    print("iteration: ", i)
    print(latent_accuracies[i])