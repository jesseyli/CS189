import numpy as np 
import scipy.io
import random
import math
import matplotlib.pyplot as plt


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
    def __init__(self, inputLayerSize = 784, hiddenLayerSize = 200, outputLayerSize = 10):
        self.inputLayerSize = inputLayerSize + 1 # add bias
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize + 1 # add bias

def trainNeuralNetwork(images,labels):
    pass
    # initialize all weights, V,W at random
    # while (stopping criteria):
    #   pick one data point at random from training set
    #   perform forward pass (compute values for gradient descent update)
    #   perform backward pass 
    #   perform stochastic gradient update
    # return V,W

def predictNeuralNetwork(weights, images):
    # compute labels of all images using the weights
    # return labels
    pass

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return np.tanh(z)

train = scipy.io.loadmat("dataset/train.mat")
train_data = train["train_images"]
train_labels= np.ravel(train["train_labels"])
train_data = train_data.reshape(-1, train_data.shape[-1])

def vectorize_labels(labels):
    new_label = np.zeros((labels.shape[0],10))
    for i in range(new_label.shape[0]):
        new_label[i][labels[i]] = 1
    return new_label

vec_labels = vectorize_labels(train_labels)

def squaredError(predictions, vec_labels):
    return np.sum((predictions-vec_labels)**2)/2

def crossEntropyError(predictions, vec_labels):
    return -np.sum(np.multiply(vec_labels,np.log(predictions)) + 
            np.multiply((1 - vec_labels), np.log(1 - predictions)))
# # Show images
# img = montage_images(train_data.reshape(28,28,60000))
# plt.imshow(img)
# plt.show()

test = scipy.io.loadmat("dataset/test.mat")
test_data = test["test_images"]