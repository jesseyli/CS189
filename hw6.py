import numpy as np 
import scipy.io
import random
import math
import matplotlib.pyplot as plt
import pickle
import pdb


def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

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

class neural_net(object):
    def __init__(self, inputLayerSize = 784, hiddenLayerSize = 200, outputLayerSize = 10, error_limit = 1e-5, lamb = 1e-2):
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.error_limit = error_limit
        self.lamb = lamb

    def squaredErrorTrain(self,images,vec_labels):
        # images = np.vstack((images, np.ones((1,images.shape[1]))))
        V = np.random.normal(0,0.01,(self.hiddenLayerSize, self.inputLayerSize + 1))
        W = np.random.normal(0,0.01,(self.outputLayerSize, self.hiddenLayerSize + 1))
        iteration = 0
        # prev_error = [float('inf')]*10
        while 10 > self.error_limit:
            sample = random.randint(0,images.shape[0] - 1)
            x = np.append(images[sample], 1) # (785,)
            XV = np.append(np.dot(x, V.T), 1) # (201,)
            h = self.tanh(XV) # (201,)
            z = self.sigmoid(np.dot(h,W.T)) # (10,)
            delta = np.multiply((z - vec_labels[sample]),self.sigmoid_prime(np.dot(h,W.T))) # (10,)
            dW = np.dot(delta.reshape((10,1)),h.reshape((1,201))) # (10,201)
            dV_first_half = np.multiply(np.dot(delta,W), self.tanh_prime(XV))[:-1]
            dV = np.dot(dV_first_half.reshape((200,1)),x.reshape(1,785))
            W = W - self.lamb*dW
            V = V - self.lamb*dV
            # delta = np.dot(np.multiply((self.sigmoid(np.dot(h,W.T)) - vec_labels[sample]),self.sigmoid_prime(np.dot(h,W.T))).T,h)
            # error = self.squaredError(self.forward((V,W),images), vec_labels)

            # error = self.squaredError(z,vec_labels[sample])
            # prev_error.pop(0)
            # prev_error.append(error)

            iteration += 1
            if iteration % 50000 == 0:
                self.lamb = self.lamb*0.9
                # Check gradients
                img = images[sample].reshape((1,784))
                eta = 1e-6
                # check_dW = (self.squaredError(self.predict((V,W+eta),img),vec_labels[sample]) 
                #             - self.squaredError(self.predict((V,W-eta),img),vec_labels[sample]))/(2*eta)
                # print('dW')
                # print(np.sum(dW))
                # print(check_dW)
                check_dV = (self.squaredError(self.predict((V+eta,W),img),vec_labels[sample]) 
                            - self.squaredError(self.predict((V-eta,W),img),vec_labels[sample]))/(2*eta)
                print('dV')
                print(np.sum(dV))
                print(check_dV)
                # pdb.set_trace()
                yield V,W
        # initialize all weights, V,W at random
        # while (stopping criteria):
        #   pick one data point at random from training set
        #   perform forward pass (compute values for gradient descent update)
        #   perform backward pass 
        #   perform stochastic gradient update
        return V,W

    def tanh(self, z):
        return np.tanh(z)

    def tanh_prime(self,z):
        return 1 - np.tanh(z)**2

    def predict(self,weights, images):
        images = np.hstack((images, np.ones((images.shape[0],1))))
        h = self.tanh(np.dot(images,weights[0].T))
        h = np.hstack((h, np.ones((h.shape[0],1))))
        z = self.sigmoid(np.dot(h,weights[1].T))
        return z
        # compute labels of all images using the weights
        # return labels

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self,z):
        return np.multiply(self.sigmoid(z), 1 - self.sigmoid(z))

    def squaredError(self,predictions, vec_labels):
        return np.sum((predictions-vec_labels)**2)/2

    def crossEntropyError(self,predictions, vec_labels):
        # a = np.log(predictions)
        # b = np.log(1 - predictions)
        error = -np.sum(np.multiply(vec_labels,np.log(predictions)) + 
                np.multiply((1 - vec_labels), np.log(1 - predictions)))
        if np.isnan(error):
            print('Taking log of zero...')
        return error

train = scipy.io.loadmat("dataset/train.mat")
train_images = train["train_images"]
train_labels= train["train_labels"]
train_images = train_images.reshape(-1, train_images.shape[-1])
train_data = np.hstack((train_images.T, train_labels))

np.random.shuffle(train_data)
train_images = train_data[:50000,:-1]
train_labels = train_data[:50000,-1]
valid_images = train_data[50000:,:-1]
valid_labels = train_data[50000:,-1]

def vectorize_labels(labels):
    new_label = np.zeros((labels.shape[0],10))
    for i in range(new_label.shape[0]):
        new_label[i][labels[i]] = 1
    return new_label

vec_labels = vectorize_labels(train_labels)

net = neural_net(lamb = 1e-2) # 1e-1 seemed to work
gen = net.squaredErrorTrain(train_images,vec_labels)
error = 1
while error > 0.03:
    V,W = gen.__next__()
    pred = net.predict((V,W),valid_images)
    error = benchmark(np.argmax(pred, axis=1),valid_labels)[0]
    print(error)
# Show images
# img = montage_images(train_images.T.reshape(28,28,50000))
# plt.imshow(img)
# plt.show()

test = scipy.io.loadmat("dataset/test.mat")
test_images = test["test_images"]
test_images = test_images.reshape(test_images.shape[0], -1)

pred = np.argmax(net.predict((V,W),test_images), axis=1)

numbers = np.arange(len(pred)) + 1
test_predict = np.vstack((numbers,pred))
np.savetxt("digits.csv", test_predict.transpose(), delimiter=",",fmt = '%u')



# # Show images
# img = montage_images(test_images.T.reshape(28,28,10000))
# plt.imshow(img)
# plt.show()

# cpickle, learning rate decreases with each iteration, gradient checking with eps = 1e-5
# examine either the magnitude of the gradient or the current cost on the whole training set every some fixed amount of iterations.
# cross validation, 10 fold?
# data preprocessing?
