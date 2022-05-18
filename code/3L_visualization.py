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
weights = np.load('3L_do_bt_weights_Hlayers120_a0.001_iters300_samples10000_batch100.npz')
weights_0_1 = weights['arr_0']
weights_1_2 = weights['arr_1']
err_count = 0
relu = lambda x: (x >= 0) * x
relu2deriv = lambda x: x >= 0
iter_data = 8
#print(weights_1_2)
for i in range(10):
    number = i
    true = np.zeros(10)
    true[number] = 1
    input = true
    print(weights_0_1)
    #wei_to_img = weights_1_2[iter_data]
    layer_0 = input
    layer_1 = relu(np.dot(layer_0.T,weights_1_2.T))
    layer_2 = np.dot(layer_1.T,weights_0_1.T)
    print(layer_2)
    for i in range(len(layer_2)):
        if layer_2[i] <= 0.0: layer_2[i] = 0.0
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))

    pred_scaling = scaler.fit_transform(layer_2.reshape(-1, 1))
    pred_scaling = pred_scaling.astype(int)
    pred_scaling_reshaped = np.reshape(pred_scaling, (len(images[0]), len(images[0][0])))

    #reverse_visualization_img = Image.fromarray(pred_scaling_reshaped, mode="L")
    #reverse_visualization_img.show()
    #print("|| true:" + str(labels[iter_data]))
    fig, ax = plt.subplots()
    #plt.gray()
    ax.imshow(pred_scaling_reshaped, cmap='gray', vmax=255, vmin=0)
    ax.set_title(str(number))
    #plt.imshow(pred_scaling_reshaped, cmap='gray', vmin=127, vmax=255)
    plt.show()

#print("pred: " + str(pred_scaling_reshaped) + "\n\n------\n")



