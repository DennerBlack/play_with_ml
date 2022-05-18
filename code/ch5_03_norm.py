import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing

# from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
n_val = 10000
images = x_test[0:n_val]
labels = y_test[0:n_val]

# print(images[0][0])

weights = np.load('ch5_my_network_stach_000.npy')

err_count = 0


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
    out = np.zeros((len(a), len(b)))

    for i in range(len(a)):
        for j in range(len(b)):
            out[i][j] = a[i] * b[j]
    return out


for iter_data in range(len(images)):
    image = np.array([np.reshape(images[iter_data], len(images[0]) * len(images[0][0]))])
    input = preprocessing.normalize(image)
    pred = neural_network(input[0], weights)
    # print(pred)
    # print(input)

    # weight_deltas = outer_prod(input[0],delta)

    print("|| true:" + str(labels[iter_data]) + " || iter:" + str(iter_data) + " || \n")
    print("pred: " + str(np.argmax(pred)) + "\n\n------\n")
    if labels[iter_data] != np.argmax(pred):
        err_count += 1
print("error percent:" + str(err_count/n_val))
