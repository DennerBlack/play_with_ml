import numpy as np

#np.random.seed(2)

def relu(x):
    return (x > 0) * x


def relu2deriv(x):
    return x > 0

alpha = 0.2
hidden_size = 10

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])

walk_vs_stop = np.array([[1, 1, 0, 0]]).T

weights_0_1 = 2*np.random.random((3, hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size, 1)) - 1

for iteration in range(300):
    error_l2 = 0
    for i in range(len(streetlights)):
        layer_1 = streetlights[i:i+1]
        layer_2 = relu(np.dot(layer_1, weights_0_1))
        layer_3 = np.dot(layer_2, weights_1_2)

        error_l2 += np.sum(np.power(layer_3 - walk_vs_stop[i:i+1], 2))

        layer_2_delta = layer_3 - walk_vs_stop[i:i+1]
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_2)

        weights_0_1 -= alpha * layer_1.T.dot(layer_1_delta)
        weights_1_2 -= alpha * layer_2.T.dot(layer_2_delta)
    if iteration % 10 == 9:
        print(("Error:" + str(error_l2)) )


print("\n\n\n\n")
'''
for iter in range(100):
    err = 0
    for i in range(len(streetlights)):
        layer_1 = streetlights[i:i+1]
        layer_2 = relu(np.dot(layer_1, weights_0_1))
        layer_3 = np.dot(layer_2, weights_1_2)

        err += np.sum(np.power(layer_3 - walk_vs_stop[i:i+1], 2))

        layer_2_delta = layer_3 - walk_vs_stop[i:i+1]
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_2)

        weights_1_2 -= alpha * layer_2.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_1.T.dot(layer_1_delta)
    if iter % 10 == 9:
        print(("Error:" + str(err)))
'''

