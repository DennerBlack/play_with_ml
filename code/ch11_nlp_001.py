import numpy as np
import sys, random
from collections import Counter
import math
from corus import load_ods_rt

path = 'rt.csv.gz'
records = load_ods_rt(path)
#a = next(records)
#print(a.text)

raw_text = ""
#print(len(records))
for txt in records:
    raw_text += " " + txt.text
    print(txt.text)

np.random.seed(1)
random.seed(1)
'''
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()
print(raw_reviews)
'''
tokens = list(map(lambda x: set(x.split(" ")), raw_text))

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


alpha = 0.05
iterations = 2
hidden_size = 50
window = 2
negative = 5

weights_0_1 = 0.2*np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0*np.random.random((len(vocab), hidden_size))

layer_2_target = np.zeros(negative+1)
layer_2_target[0] = 1

correct = 0
total = 0

for rev_i, review in enumerate(input_dataset * iterations):
    for target_i in range(len(review)):
        target_samples = [review[target_i]] + list(concatenated[(np.random.rand(negative)*len(concatenated)).astype('int').tolist()])

        left_context = review[max(0, target_i - window):target_i]
        right_context = review[target_i+1:min(len(review), target_i + window)]

        layer_1 = np.mean(weights_0_1[left_context+right_context], axis=0)
        layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))

        layer_2_delta = layer_2 - layer_2_target
        layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])

        weights_0_1[left_context+right_context] -= alpha * layer_1_delta
        weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha
    if (rev_i % 250 == 0):
        sys.stdout.write('\rProgress:' + str(rev_i / float(len(input_dataset)
                                                           * iterations)) + "   " + str(similar('terrible')))
    sys.stdout.write('\rProgress:' + str(rev_i / float(len(input_dataset)
                                                       * iterations)))
print(similar('terrible'))

file_name = "NLP0_analogy_weights_Hlayers" + str(hidden_size) + "_a" + str(alpha) + \
            "_iters" + str(iterations)

np.savez(file_name, weights_0_1, weights_1_2)



