from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

mnist = MNIST('C:/Users/farsh/Documents/MATLAB/Github/MNIST Dataset/')
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