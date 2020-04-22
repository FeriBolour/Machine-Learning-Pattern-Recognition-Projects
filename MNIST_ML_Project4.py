from mnist import MNIST
import numpy as np
from scipy.stats import zscore
#import matplotlib.pyplot as plt

def softmax(X,Wj,W):
    e_x = np.exp(np.dot(X,Wj))
    return e_x/np.sum(np.exp(np.dot(X,W)),axis =1)

def getGrad(x_train,T,W):
    grad = np.zeros([T.shape[1],1])
    for j in range(T.shape[1]):
        for i in range(x_train.shape[0]):
            Y = softmax(x_train[i,:],W[:,j],W)
            #grad[j] += np.dot
    return grad


mnist = MNIST('V:/Machine-Learning-Pattern-Recognition-Projects/MNIST Dataset')
x_train, y_train = mnist.load_training() #60000 samples
x_test, y_test = mnist.load_testing()    #10000 samples

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)

#print(x_train.shape)

#img1_arr, img1_label = x_train[1], y_train[1]
#print(img1_arr.shape, img1_label)

# reshape first image(1 D vector) to 2D dimension image
#img1_2d = np.reshape(img1_arr, (28, 28))
# show it
#plt.subplot(111)
#plt.imshow(img1_2d, cmap=plt.get_cmap('gray'))
#plt.show()

x_train= zscore(x_train,axis=0)

# one-hot encoding
T = np.zeros([len(y_train),len(np.unique(y_train))])

for i in range(len(y_train)):
    T[i,y_train[i]] = 1
    
#W's offset
x_train = np.column_stack((x_train,np.ones(len(x_train),)))

W = np.random.normal(size = (x_train.shape[1],len(np.unique(y_train))))

beta = 0.9
learning_rate = 1e-5

getGrad(x_train,T,W)


    