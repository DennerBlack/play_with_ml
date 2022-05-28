import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt



fl="3L_sm_th_weights_Hlayers100_a2_iters300_samples1000_batch100.npz"
weights = np.load(fl)
weights_0_1 = weights['arr_0']
weights_1_2 = weights['arr_1']
err_count = 0

def tanh(x):
    return np.tanh(x)

def tanh2deriv(output):
    return 1 - (np.power(output, 2))

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, keepdims=True)

scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))

for i in range(10):
    true = np.zeros(10)
    true[i] = 1
    layer_0 = true
    layer_1 = tanh(np.dot(layer_0.T,weights_1_2.T))
    #layer_2 = softmax(np.dot(layer_1.T,weights_0_1.T))
    layer_2 = np.dot(layer_1.T,weights_0_1.T)
    print(weights_0_1.shape)
    test = np.dot(weights_0_1, weights_1_2)
    print(test.shape)
    pred_scaling = scaler.fit_transform(test.T[i].reshape(-1, 1))
    pred_scaling = pred_scaling.astype(int)
    pred_scaling_reshaped = np.reshape(pred_scaling, (28, 28))

    fig, ax = plt.subplots()
    ax.imshow(pred_scaling_reshaped, cmap='gray', vmax=255, vmin=0)
    ax.set_title(str(i))
    plt.show()





