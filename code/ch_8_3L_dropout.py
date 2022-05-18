import numpy as np
import sys
from tensorflow.keras.datasets import mnist

n_samples = 10000

(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:n_samples].reshape(n_samples, 28 * 28) / 255, y_train[0:n_samples])

one_hot_labels = np.zeros((len(labels), 10))
#print(len(y_test))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)

relu = lambda x: (x >= 0) * x
relu2deriv = lambda x: x >= 0

alpha = 0.005
iterations = 300
hidden_size = 100
pixel_per_image = 784
num_labels = 10

weights_0_1 = 0.2 * np.random.random((pixel_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    err = 0.0
    correct_cnt = 0
    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot(layer_1, weights_1_2)

        err += np.sum(np.power(layer_2- labels[i:i+1], 2))
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = layer_2 - labels[i:i+1]
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
    sys.stdout.write("\r I:" + str(j) + \
                     " Train-Err:" + str(err / float(len(images)))[0:5] + \
                     " Train-Acc:" + str(correct_cnt / float(len(images))))


file_name = "3L_dropout_weights_Hlayers" + str(hidden_size) + "_a" + str(alpha) + "_iters" + str(iterations) + "_samples" + str(n_samples)

np.savez(file_name, weights_0_1, weights_1_2)

correct_cnt = 0
for i in range(len(test_images)):
    layer_0 = test_images[i:i+1]
    layer_1 = relu(np.dot(layer_0,weights_0_1))
    layer_2 = np.dot(layer_1,weights_1_2)
    correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))
    print("true: " + str(np.argmax(test_labels[i:i+1])) + "  ||  pred: " + str(np.argmax(layer_2)))
sys.stdout.write(" Test-Acc:" + str(correct_cnt/float(len(test_images))) + "\n")
print()