import numpy as np
import sys
from tensorflow.keras.datasets import mnist
#from keras.datasets import mnist
np.set_printoptions(threshold=sys.maxsize)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[0:1000]
labels = y_train[0:1000]

print(images[0][0])


weights = np.random.rand(10, len(images[0][0])*len(images[0][0]))

def w_sum(a,b):
    assert(len(a) == len(b))
    output = 0

    for i in range(len(a)):
        output += (a[i] * b[i])

    return output

def vect_mat_mul(vect,matrix):
    assert(len(vect) == len(matrix[0]))
    output = np.zeros(len(matrix))
    for i in range(len(matrix)):
        output[i] = w_sum(vect,matrix[i])
    return output

def neural_network(input, weights):
    pred = vect_mat_mul(input,weights) #input.dot(weights)
    return pred

def outer_prod(a, b):
    # just a matrix of zeros
    out = np.zeros((len(a), len(b)))

    for i in range(len(a)):
        for j in range(len(b)):
            out[i][j] = a[i] * b[j]
    return out

alpha = 0.00000001

for iter_data in range(len(images)):
    input = np.array([np.reshape(images[iter_data],len(images[0])*len(images[0][0]))])
    true = np.zeros(10)
    true[labels[iter_data]] = 1000
    error = np.array([0,0,0,0,0,0,0,0,0,0])#np.array([0]*len(images[iter_data])**2)
    delta = np.array([0,0,0,0,0,0,0,0,0,0])#np.array([0]*len(images[iter_data])**2)
    print("start loop")
    for k in range(150):
        pred = neural_network(input[0],weights)
        #print(pred)
        #print(input)
        for i in range(len(pred)):
            error[i] = (pred[i] - true[i]) ** 2
            delta[i] = pred[i] - true[i]
    
        weight_deltas = outer_prod(delta, input[0])
        #print(weight_deltas)
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                weights[i][j] -= alpha * weight_deltas[i][j]
    
        #weight_deltas = outer_prod(input[0],delta)
        #print(weight_deltas)
        if k == 0 or k == 149 or k == 49 or k == 99:
            print("|| true:"+str(labels[iter_data])+" || iter:"+str(iter_data)+" || loop:"+ str(k)+" || \n")
            print("pred: " + str(pred) + "\n\n------\n")
    print("end loop \n\n############################\n")

np.save('ch5_my_network_002', weights)
np.save('network_data/ch5_my_network_002', weights)
