import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


###     Settings      ###
n_samples = 10000
batch_size = 128
alpha = 2
iterations = 20
pixel_per_image = 784
num_labels = 10

input_rows = 28
input_cols = 28
kernel_rows = 3
kernel_cols = 3
num_kernels = 16

hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

###       Code        ###



def tanh(x):
    return np.tanh(x)


def tanh2deriv(output):
    return 1 - (np.power(output, 2))


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


fl="conv1_weights_Hlayers7056_a2_iters20_samples10000_batch128_ksize7.npz"
weights = np.load(fl)
kernels = weights['arr_0']
weights_1_2 = weights['arr_1']
err_count = 0


def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)


scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))

for i in range(16):
    '''
    true = np.zeros(10)
    true[i] = 1
    
    layer_0 = true
    
    layer_1 = np.dot(layer_0.T, weights_1_2.T)


    layer_2 = layer_1.reshape(16, 25, 25)
    layer_2.shape

    sects = list()
    for row_start in range(layer_2.shape[1] - kernel_rows):
        for col_start in range(layer_2.shape[2] - kernel_cols):
            sect = get_image_section(layer_2,
                                     row_start,
                                     row_start + kernel_rows,
                                     col_start,
                                     col_start + kernel_cols)
            sects.append(sect)

    expanded_input = np.concatenate(sects, axis=1)
    es = expanded_input.shape
    flattened_input = expanded_input.reshape(es[0] * es[1], -1)

    kernel_output = flattened_input.dot(kernels)
    layer_2 = tanh(kernel_output.reshape(es[0], -1))
    l2_reshaped = np.reshape(layer_2, (256, 22, 22))
    l2_con = np.zeros((22,22))#np.concatenate(l2_reshaped, axis=2)'''
    '''for k in range(22):
        for j in range(22):
            temp = 0
            for l in range(256):
                temp+= l2_reshaped[l][j][k]
            temp /= 256
            l2_con[j][k] = temp'''
    ker_res = np.reshape(kernels, (16, 49))
    #print(np.reshape(kernels, (16, 49)))

    pred_scaling = scaler.fit_transform(ker_res[i].reshape(-1, 1))
    pred_scaling = pred_scaling.astype(int)
    pred_scaling_reshaped = np.reshape(pred_scaling, (7, 7))

    fig, ax = plt.subplots()
    ax.imshow(pred_scaling_reshaped, cmap='gray', vmax=255, vmin=0)
    ax.set_title(str(i))
    plt.show()


