import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import math, random
import scipy.io



# Problem 3
'''
X1 = np.random.normal(3,3,100)
X2 = (X1/2) + np.random.normal(4,2,100)

# Part (a)
mu = [np.sum(X1)/100, np.sum(X2)/100]
print('mean:\n', mu)

# Part (b)
sigma = np.cov(np.vstack((X1,X2)))
print('covariance matrix:\n',sigma)

# Part (c)
eigen =  np.linalg.eig(sigma)
eigvalues = eigen[0]
eigvectors = eigen[1].T
print('eigenvalues:\n',eigvalues)
print('eigenvectors:\n',eigvectors)

# Part (d)
plt.figure()
plt.plot(X1,X2,'.')
plt.xlim(-15,15)
plt.ylim(-15,15) 
U,V = zip(*eigvectors)
U = np.multiply(U,eigvalues)
V = np.multiply(V,eigvalues)
ax = plt.gca()
ax.quiver(mu[0], mu[1], U,V,scale_units='xy',scale=1)

plt.title('part d')
plt.xlabel("X1")
plt.ylabel("X2")

# Part (e)

if eigvalues[0] > eigvalues[1]:
    U = np.array(eigvectors).T
else:
    U = np.array([eigvectors[1],eigvalues[0]]).T

rotated = np.dot(U.T,(np.vstack([X1,X2]) - np.tile(mu,[100,1]).T))

plt.figure()
plt.plot(rotated[0],rotated[1],'.')
plt.xlim(-15,15)
plt.ylim(-15,15) 
plt.title('part e')
plt.show()

'''


# Problem 5

#benchmark.m, converted
def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

#montage_images.m, converted
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

digit_train_data = scipy.io.loadmat("data/digit_dataset/train.mat")
digit_train_images= digit_train_data["train_images"]
digit_train_labels= digit_train_data["train_labels"]

train_vectors=[]
for i in range(np.shape(digit_train_images)[2]):
    vector = np.float64(digit_train_images[:,:,i].flatten())
    train_vectors.append( np.divide(vector,np.linalg.norm(vector, 2)))
normalized_vectors= np.array(train_vectors)


indices = list(range(np.shape(normalized_vectors)[0]))
random.shuffle(indices)
train_vectors = []
train_labels = []
val_vectors = []
val_labels = []
for i in indices[:50000]:
    train_vectors.append(normalized_vectors[i])
    train_labels.append(digit_train_labels[i])
for i in indices[50000:]:
    val_vectors.append(normalized_vectors[i])
    val_labels.append(digit_train_labels[i])
val_labels = np.array(val_labels).ravel()
train_labels = np.array(train_labels).ravel()
train_vectors = np.array(train_vectors)
val_vectors = np.array(val_vectors)

# part (a)
'''
mean_list = []
cov_list = []
for i in range(10):
    indices = np.where(digit_train_labels==i)[0]
    subset = normalized_vectors[indices]
    mean_list.append(np.sum(subset,axis=0)/subset.shape[0])
    cov_list.append(np.cov(subset.T))

# mean_list and cov_list are the mean and covariance matrices for part (a)
# mean_list[i] and cov_list[i] correspond to digit i
'''

# part (b)
'''
priors = []
for i in range(10):
    indices = np.where(digit_train_labels==i)[0]
    priors.append(indices.shape[0]/digit_train_labels.shape[0])
print('Priors:\n', priors)
'''

# part (c)

'''
plt.imshow(cov_list[1])
plt.colorbar()
plt.show()
'''

# part (di)

'''
training_size = [100,200,500,1000,2000,5000,10000,30000,50000]
errors = []
alpha = np.eye(784)*0.0001
for N in training_size:

    subset_vectors = train_vectors[:N]
    subset_labels = train_labels[:N]
    
    mean_list = []
    cov_list = []
    for i in range(10):
        indices = np.where(subset_labels==i)[0]
        subset = subset_vectors[indices]
        mean_list.append(np.sum(subset,axis=0)/subset.shape[0])
        cov_list.append(np.cov(subset.T))

    priors = []
    for i in range(10):
        indices = np.where(subset_labels==i)[0]
        priors.append(indices.shape[0]/subset_labels.shape[0])

    sigma_overall = np.sum(cov_list,axis=0)/len(cov_list) + alpha
    sig_inv = np.linalg.inv(sigma_overall)
    predictions= []
    for sample in val_vectors:
        guess = 0
        best = -float('inf')
        for i in range(10):
            f = (np.dot(np.dot(mean_list[i].T,sig_inv),sample) - np.dot(np.dot(mean_list[i].T,sig_inv),mean_list[i])/2 + math.log(priors[i]))
            if f > best:
                guess = i
                best = f
        predictions.append(guess)
    errors.append(benchmark(np.array(predictions),val_labels)[0])

plt.figure()
plt.plot(training_size,errors,'-')
plt.show()
'''

# part (dii)

'''
training_size = [100,200,500,1000,2000,5000,10000,30000,50000]
errors = []
alpha = np.eye(784)*0.00001
for N in training_size:
    randomidx = random.randint(0,50000-N)
    subset_vectors = train_vectors[randomidx:randomidx+N]
    subset_labels = train_labels[randomidx:randomidx+N]
    
    mean_list = []
    cov_list = []
    for i in range(10):
        indices = np.where(subset_labels==i)[0]
        subset = subset_vectors[indices]
        mean_list.append(np.sum(subset,axis=0)/subset.shape[0])
        cov_list.append(np.cov(subset.T) + alpha)

    priors = []
    for i in range(10):
        indices = np.where(subset_labels==i)[0]
        priors.append(indices.shape[0]/subset_labels.shape[0])

    sig_inv = [np.linalg.inv(sigma) for sigma in cov_list]

    logdet_list = [np.linalg.slogdet(sigma)[1] for sigma in cov_list] 
    predictions= []
    for sample in val_vectors:
        guess = 0
        best = -float('inf')
        for i in range(10):
            f = np.dot(np.dot((sample - mean_list[i]).T,sig_inv[i]),(sample - mean_list[i]))/-2 + math.log(priors[i]) - logdet_list[i]/2
            if f > best:
                guess = i
                best = f
        predictions.append(guess)
    errors.append(benchmark(np.array(predictions),val_labels)[0])

plt.figure()
plt.plot(training_size,errors,'-')
plt.show()
'''

# part (div)

'''
# LDA for part (div)
digit_test_data = scipy.io.loadmat("data/digit_dataset/test.mat")
digit_test_images= digit_test_data["test_images"].transpose()

test_vectors=[]
for i in range(np.shape(digit_test_images)[1]):
    vector = np.float64(digit_test_images[:,i])
    vector = vector.reshape((28,28)).T
    vector = vector.flatten()
    test_vectors.append(np.divide(vector,np.linalg.norm(vector, 2)))
test_vectors= np.array(test_vectors)

alpha = np.eye(784)*0.0001

subset_vectors = train_vectors
subset_labels = train_labels

mean_list = []
cov_list = []
for i in range(10):
    indices = np.where(subset_labels==i)[0]
    subset = subset_vectors[indices]
    mean_list.append(np.sum(subset,axis=0)/subset.shape[0])
    cov_list.append(np.cov(subset.T))

priors = []
for i in range(10):
    indices = np.where(subset_labels==i)[0]
    priors.append(indices.shape[0]/subset_labels.shape[0])

sigma_overall = np.sum(cov_list,axis=0)/len(cov_list) + alpha
sig_inv = np.linalg.inv(sigma_overall)
predictions= []
for sample in val_vectors:
    guess = 0
    best = -float('inf')
    for i in range(10):
        f = (np.dot(np.dot(mean_list[i].T,sig_inv),sample) - np.dot(np.dot(mean_list[i].T,sig_inv),mean_list[i])/2 + math.log(priors[i]))
        if f > best:
            guess = i
            best = f
    predictions.append(guess)
error = benchmark(np.array(predictions),val_labels)[0]


test_predict = []
for sample in test_vectors:
    guess = 0
    best = -float('inf')
    for i in range(10):
        f = (np.dot(np.dot(mean_list[i].T,sig_inv),sample) - np.dot(np.dot(mean_list[i].T,sig_inv),mean_list[i])/2 + math.log(priors[i]))
        if f > best:
            guess = i
            best = f
    test_predict.append(guess)


numbers = (np.arange(10000) + 1)
print(test_predict[:25])
test_predict = np.vstack((numbers,test_predict))
print(error)
np.savetxt("digitsLDA.csv", test_predict.transpose(), delimiter=",",fmt = '%u')

img = montage_images(test_vectors.T.reshape((28,28,test_vectors.shape[0]))[:,:,:25])
plt.imshow(img)
plt.show()
'''


# QDA for part (div)
'''
digit_test_data = scipy.io.loadmat("data/digit_dataset/test.mat")
digit_test_images= digit_test_data["test_images"].transpose()



test_vectors=[]
for i in range(np.shape(digit_test_images)[1]):
    vector = np.float64(digit_test_images[:,i])
    vector = vector.reshape((28,28)).T
    vector = vector.flatten()
    test_vectors.append(np.divide(vector,np.linalg.norm(vector, 2)))
test_vectors= np.array(test_vectors)


alpha = np.eye(784)*0.00001


mean_list = []
cov_list = []
for i in range(10):
    indices = np.where(train_labels==i)[0]
    subset = train_vectors[indices]
    mean_list.append(np.sum(subset,axis=0)/subset.shape[0])
    cov_list.append(np.cov(subset.T) + alpha)

priors = []
for i in range(10):
    indices = np.where(train_labels==i)[0]
    priors.append(indices.shape[0]/train_labels.shape[0])

sig_inv = [np.linalg.inv(sigma) for sigma in cov_list]

logdet_list = [np.linalg.slogdet(sigma)[1] for sigma in cov_list] 
validation_predict = []
for sample in val_vectors:
    guess = 0
    best = -float('inf')
    for i in range(10):
        f = np.dot(np.dot((sample - mean_list[i]).T,sig_inv[i]),(sample - mean_list[i]))/-2 + math.log(priors[i]) - logdet_list[i]/2
        if f > best:
            guess = i
            best = f
    validation_predict.append(guess)
error = benchmark(np.array(validation_predict),val_labels)[0]
test_predict = []
for sample in test_vectors:
    guess = 0
    best = -float('inf')
    for i in range(10):
        f = np.dot(np.dot((sample - mean_list[i]).T,sig_inv[i]),(sample - mean_list[i]))/-2 + math.log(priors[i]) - logdet_list[i]/2
        if f > best:
            guess = i
            best = f
    test_predict.append(guess)


numbers = (np.arange(10000) + 1)
print(test_predict[:25])
test_predict = np.vstack((numbers,test_predict))
print(benchmark(validation_predict, val_labels)[0])
np.savetxt("digitsQDA.csv", test_predict.transpose(), delimiter=",",fmt = '%u')

img = montage_images(test_vectors.T.reshape((28,28,test_vectors.shape[0]))[:,:,:25])
plt.imshow(img)
plt.show()
'''



# part (e)
'''
spam_data = scipy.io.loadmat("data/spam_dataset/spam_data.mat")
spam_train_data= spam_data["training_data"]
spam_train_labels= np.ravel(spam_data["training_labels"])
spam_test = spam_data["test_data"]


normalize = []
for vector in spam_test:
    if not np.linalg.norm(vector, 2):
        normalize.append(vector)
    else:
        normalize.append(np.divide(vector,np.linalg.norm(vector, 2)))
spam_test = np.array(normalize)

normalize = []
for vector in spam_train_data:
    if not np.linalg.norm(vector, 2):
        normalize.append(vector)
    else:
        normalize.append(np.divide(vector,np.linalg.norm(vector, 2)))
spam_train_data = np.array(normalize)

alpha = np.eye(32)*0.0001
indices = list(range(np.shape(spam_train_data)[0]))
random.shuffle(indices)

subset_vectors = []
subset_labels = []
val_vectors = []
val_labels = []
for i in indices[:4000]:
    subset_vectors.append(spam_train_data[i])
    subset_labels.append(spam_train_labels[i])
for i in indices[4000:]:
    val_vectors.append(spam_train_data[i])
    val_labels.append(spam_train_labels[i])
val_labels = np.array(val_labels).ravel()
subset_labels = np.array(subset_labels).ravel()
subset_vectors = np.array(subset_vectors)
val_vectors = np.array(val_vectors)

mean_list = []
cov_list = []
priors = []
for i in range(2):
    indices = np.where(subset_labels==i)[0]
    subset = subset_vectors[indices]
    mean_list.append(np.sum(subset,axis=0)/subset.shape[0])
    cov_list.append(np.cov(subset.T) + alpha)
    priors.append(indices.shape[0]/subset_labels.shape[0])

sig_inv = [np.linalg.inv(sigma) for sigma in cov_list]

logdet_list = [np.linalg.slogdet(sigma)[1] for sigma in cov_list] 
predictions= []
for sample in val_vectors:
    guess = 0
    best = -float('inf')
    for i in range(2):
        f = np.dot(np.dot((sample - mean_list[i]).T,sig_inv[i]),(sample - mean_list[i]))/-2 + math.log(priors[i]) - logdet_list[i]/2
        if f > best:
            guess = i
            best = f
    predictions.append(guess)
error = benchmark(np.array(predictions),val_labels)[0]

test_predict = []
for sample in spam_test:
    guess = 0
    best = -float('inf')
    for i in range(2):
        f = np.dot(np.dot((sample - mean_list[i]).T,sig_inv[i]),(sample - mean_list[i]))/-2 + math.log(priors[i]) - logdet_list[i]/2
        if f > best:
            guess = i
            best = f
    test_predict.append(guess)


numbers = (np.arange(5857) + 1)
print('val error: ',error)
test_predict = np.vstack((numbers,test_predict))
np.savetxt("spam.csv", test_predict.transpose(), delimiter=",",fmt = '%u')
'''

# problem 6
'''
housing_data = scipy.io.loadmat("data/housing_dataset/housing_data.mat")
X_train= housing_data['Xtrain']
X = np.hstack((X_train,np.ones((X_train.shape[0],1))))
Y_train= housing_data['Ytrain']
y = np.float_(Y_train)[:,0]

w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))

X_validate = housing_data['Xvalidate']
X_validate = np.hstack((X_validate,np.ones((X_validate.shape[0],1))))
Y_validate= housing_data['Yvalidate'][:,0]

validation_predict = np.dot(X_validate,w)
RSS = np.dot((validation_predict - Y_validate).T,(validation_predict - Y_validate))
print('RSS: ',RSS)
print('predicted min: ',min(validation_predict))
print('predicted max: ',max(validation_predict))

plt.figure()
plt.plot(np.arange(8),w[:8],'o')
plt.xlim((-1,9))

plt.figure()
plt.hist(np.dot(X,w) - y,30)
plt.show()
'''