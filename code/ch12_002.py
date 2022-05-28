import numpy as np


def softmax(x_):
    x = np.atleast_2d(x_)
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


word_vects = {}
word_vects['yankees'] = np.array([[0., 0., 0.]])
word_vects['bears'] = np.array([[0., 0., 0.]])
word_vects['braves'] = np.array([[0., 0., 0.]])
word_vects['red'] = np.array([[0., 0., 0.]])
word_vects['sox'] = np.array([[0., 0., 0.]])
word_vects['lose'] = np.array([[0., 0., 0.]])
word_vects['defeat'] = np.array([[0., 0., 0.]])
word_vects['beat'] = np.array([[0., 0., 0.]])
word_vects['tie'] = np.array([[0., 0., 0.]])

sent2output = np.random.rand(3,len(word_vects))
print(f"{sent2output=}")
identity = np.eye(3)

layer_0 = word_vects['red']
layer_1 = layer_0.dot(identity) + word_vects['sox']
layer_2 = layer_1.dot(identity) + word_vects['defeat']
print(f"{layer_0=}")
print(f"{layer_1=}")
print(f"{layer_2=}")
print(f"{layer_2.dot(sent2output)=}")
pred = softmax(layer_2.dot(sent2output))
print(pred)

