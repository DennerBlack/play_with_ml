import numpy as np
import sys, random
from collections import Counter
import math

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
vocab = list(set(map(lambda x: x[0], wordcnt.most_common())))

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i


def similar(target='beautiful'):
    target_index = word2index[target]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - weights_0_1[target_index]
        squared_diff = raw_difference ** 2
        scores[word] = -math.sqrt(sum(squared_diff))
    return scores.most_common(10)


def analogy(positive=['terrible', 'good'], negative=['bad']):
    norms = np.sum(weights_0_1 * weights_0_1, axis=1)
    norms.resize(norms.shape[0], 1)
    normed_weights = weights_0_1 * norms
    query_vect = np.zeros(len(weights_0_1[0]))
    for word in positive:
        query_vect += normed_weights[word2index[word]]
    for word in negative:
        query_vect -= normed_weights[word2index[word]]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - query_vect
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return [positive, negative, scores.most_common(10)[1:]]

goal = analogy(["man"], ["dick"])
print(str(goal[0])+" - " + str(goal[1]) + "~=\n" + str(goal[2]))
