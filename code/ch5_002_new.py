import numpy as np
import sys
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing

# from keras.datasets import mnist
np.set_printoptions(threshold=sys.maxsize)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[0:1000]
labels = y_train[0:1000]

print(images[0][0])

weights = 2*np.random.rand(10, len(images[0][0]) * len(images[0][0]))-1


def w_sum(a, b):
    assert (len(a) == len(b))
    output = 0

    for i in range(len(a)):
        output += (a[i] * b[i])

    return output


def vect_mat_mul(vect, matrix):
    assert (len(vect) == len(matrix[0]))
    output = np.zeros(len(matrix))
    for i in range(len(matrix)):
        output[i] = w_sum(vect, matrix[i])
    return output


def neural_network(input, weights):
    pred = vect_mat_mul(input, weights)  # input.dot(weights)
    return pred


def outer_prod(a, b):
    # just a matrix of zeros
    out = np.zeros((len(a), len(b)))

    for i in range(len(a)):
        for j in range(len(b)):
            out[i][j] = a[i] * b[j]
    return out


alpha = 0.1
for k in range(150):
    err_global = 0.0
    for iter_data in range(len(images)):
        image = np.array([np.reshape(images[iter_data], len(images[0]) * len(images[0][0]))])
        input = preprocessing.normalize(image)
        true = np.zeros(10)
        #print(true)
        true[labels[iter_data]] = 1
        error = np.zeros(10)  # np.array([0]*len(images[iter_data])**2)
        delta = np.zeros(10)  # np.array([0]*len(images[iter_data])**2)


        pred = neural_network(input[0], weights)
        # print(pred)
        # print(input)
        for i in range(len(pred)):
            error[i] = np.power(pred[i] - true[i], 2)
            #print(error[i])
            err_global += error[i]
            delta[i] = pred[i] - true[i]

        weight_deltas = outer_prod(delta, input[0])
        # print(weight_deltas)
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                weights[i][j] -= alpha * weight_deltas[i][j]

        # weight_deltas = outer_prod(input[0],delta)
        # print(weight_deltas)
        #if k == 99 or k == 0 or k == 49:
        print("|| true:" + str(labels[iter_data]) + " || iter:" + str(iter_data) + " || loop:" + str(k) + " || \n")
        print("pred: " + str(pred) + "\n\n------\n")
    print("global error = " + str(err_global))

np.save('ch5_my_network_stach_000', weights)
np.save('network_data/ch5_my_network_stach_000', weights)
