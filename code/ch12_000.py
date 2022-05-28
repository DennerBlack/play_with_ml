import numpy as np
import sys, random
from collections import Counter
import math
import matplotlib.pyplot as plt

fl = "NLP0_analogy_weights_Hlayers50_a0.05_iters2.npz"
weights = np.load(fl)
weights_0_1 = weights['arr_0']
weights_1_2 = weights['arr_1']

np.random.seed(1)
random.seed(1)
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

tokens = list(map(lambda x: set(x.split(" ")), raw_reviews))

wordcnt = Counter()
for sent in tokens:
    for word in sent:
        wordcnt[word] -= 1
vocab = list(set(map(lambda x:x[0],wordcnt.most_common())))
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

concatenated = list()
input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
            concatenated.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sent_indices)))
concatenated = np.array(concatenated)
random.shuffle(input_dataset)

target_dataset = list()


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid2deriv(x):
    return x*(1-x)

def tanh(x):
    return np.tanh(x)


def tanh2deriv(output):
    return 1 - (np.power(output, 2))

def similar(target='beautiful'):
    target_index = word2index[target]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - weights_0_1[target_index]
        squared_diff = raw_difference**2
        scores[word] = -math.sqrt(sum(squared_diff))
    return scores.most_common(10)

norms = np.sum(weights_0_1*weights_0_1,axis=1)
norms.resize(norms.shape[0],1)
normed_weights = weights_0_1 * norms



def make_sent_vect(words):
    indices = list(map(lambda x:word2index[x], filter(lambda x:x in word2index, words)))
    return np.mean(normed_weights[indices],axis=0)

reviews2vectors = list()
for reviews in tokens:
    reviews2vectors.append(make_sent_vect(reviews))
reviews2vectors = np.array(reviews2vectors)

'''
print(reviews2vectors[0])
plt.plot(reviews2vectors[0])
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel('normed_weights')
plt.show()
'''

def most_similar_reviews(review):
    v = make_sent_vect(review)

    #print(v)
    plt.plot(v)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.ylabel('normed_weights')
    plt.show()

    scores = Counter()
    for i, val in enumerate(reviews2vectors.dot(v)):
        '''print(val)
        plt.plot(val)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.ylabel('normed_weights')
        plt.show()'''
        scores[i] = val
    most_similar = list()

    for idx, score in scores.most_common(3):
        most_similar.append(raw_reviews[idx][0:100])
    return most_similar


print(1,most_similar_reviews(["wonderful", 'terrible', 'boring', 'bad', 'cool', 'strong']))
print(2,most_similar_reviews(["wonderful", 'terrible', 'boring', 'bad', 'cool']))
print(3,most_similar_reviews(["wonderful", 'terrible', 'boring', 'bad']))
print(4,most_similar_reviews(["wonderful", 'terrible', 'boring']))
print(5,most_similar_reviews(["wonderful", 'terrible']))
print(6,most_similar_reviews(["wonderful"]))

print(most_similar_reviews(["wonderful", 'terrible', 'hi', 'lol', 'hello', 'sky', 'impossible', 'cut'
                            , 'what', 'keep', 'awful', 'danger', 'happy', 'smile', 'strong', 'book',
                            'defeat', 'kill', 'boring', 'minimal', 'bad', 'early', 'job', 'marry', "wonderful", 'terrible', 'hi', 'lol', 'hello', 'sky', 'impossible', 'cut'
                            , 'what', 'keep', 'awful', 'danger', 'happy', 'smile', 'strong', 'book',
                            'defeat', 'kill', 'boring', 'minimal', 'bad', 'early', 'job', 'marry']))

alpha = 0.05
iterations = 2
hidden_size = 50
window = 2
negative = 5