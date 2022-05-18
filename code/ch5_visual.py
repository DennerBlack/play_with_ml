import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing
from PIL import Image
import matplotlib.pyplot as plt
# from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
n_val = 10000
images = x_test[0:n_val]
labels = y_test[0:n_val]

# print(images[0][0])

weights = np.load('ch5_my_network_003.npy')

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


iter_data = 8


wei_to_img = weights[iter_data]
#for i in range(len(wei_to_img)):
#    if wei_to_img[i] <= 0.0: wei_to_img[i] = 0.0
scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))

pred_scaling = scaler.fit_transform(wei_to_img.reshape(-1, 1))
pred_scaling = pred_scaling.astype(int)
pred_scaling_reshaped = np.reshape(pred_scaling, (len(images[0]), len(images[0][0])))

#reverse_visualization_img = Image.fromarray(pred_scaling_reshaped, mode="L")
#reverse_visualization_img.show()
print("|| true:" + str(labels[iter_data]))
fig, ax = plt.subplots()
#plt.gray()
ax.imshow(pred_scaling_reshaped, cmap='gray', vmax=255, vmin=0)
#plt.imshow(pred_scaling_reshaped, cmap='gray', vmin=127, vmax=255)
plt.show()

#print("pred: " + str(pred_scaling_reshaped) + "\n\n------\n")



