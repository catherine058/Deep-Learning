"""
Stencil layout for your trigram language model assignment, with embeddings.

The stencil has three main parts:
    - A class stencil for your actual trigram language model. The class is complete with helper
    methods declarations that break down what your model needs to do into reasonably-sized steps,
    which we recommend you use.

    - A "read" helper function to isolate the logic of parsing the raw text files. This is where
    you should build your vocabulary, and transform your input files into data that can be fed into the model.

    - A main-training-block area - this code (under "if __name__==__main__") will be run when the script is,
    so it's where you should bring everything together and execute the actual training of your model.
"""

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class TrigramLM:
    def __init__(self, X1, X2, Y, vocab_sz):
        """
        Instantiate your TrigramLM Model, with whatever hyperparameters are necessary
        !!!! DO NOT change the parameters of this constructor !!!!

        X1, X2, and Y represent the first, second, and third words of a batch of trigrams.
        (hint: they should be placeholders that can be fed batches via your feed_dict).

        You should implement and use the "read" function to calculate the vocab size of your corpus
        before instantiating the model, as the model should be specific to your corpus.
        """
        
        # TODO: Define network parameters

        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.vocab_sz = vocab_sz
 
        self.logits = self.forward_pass()

        # IMPORTANT - your model MUST contain two instance variables,
        # self.loss and self.train_op, that contain the loss computation graph 
        # (as you will define in in loss()), and training operation (as you will define in train())
        self.loss = self.loss()
        self.train_op = self.optimizer()

    def forward_pass(self):
        """
        Build the inference computation graph for the model, going from the input to the output
        logits (before final softmax activation). This is analogous to "prediction".
        """

        # TODO: Compute the logits
        X1 = self.X1
        X2 = self.X2
        Y = self.Y
        EMBEDDING_SZ = 30
        # TODO: Create Network Weights
        E = tf.Variable(tf.random_normal([self.vocab_sz, EMBEDDING_SZ], stddev=0.1))
        W = tf.Variable(tf.truncated_normal([700, self.vocab_sz],
                                            stddev=0.1))
        b = tf.Variable(tf.zeros([self.vocab_sz]))

        # TODO: Build Inference Pipeline
        embedding = tf.nn.embedding_lookup(E, X1)
        embedding2 = tf.nn.embedding_lookup(E, X2)
        both = tf.concat([embedding, embedding2], 1)
        hiddenw = tf.Variable(tf.truncated_normal([EMBEDDING_SZ*2, 700],
                                            stddev=0.1))
        h_b = tf.Variable(tf.zeros([700]))
        h_l = tf.nn.relu(tf.matmul(both, hiddenw) + h_b)
        logits = tf.matmul(h_l, W) + b
        return logits

    def loss(self):
        """
        Build the cross-entropy loss computation graph.
        DO 
        """

        # TODO: Perform the loss computation
        xent = tf.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.Y)
        loss = tf.reduce_sum(xent)
        return loss

    def optimizer(self):
        """
        Build the training operation, using the cross-entropy loss and an Adam Optimizer.
        """

        # TODO: Execute the training operation

        return tf.train.AdamOptimizer(1e-4).minimize(self.loss)


def read(train_file, dev_file):
    """
    Read and parse the file, building the vectorized representations of the input and output.
    :param train_file: Path to the training file.
    :param dev_file: Path to the test file.
    """

    # TODO: Read and process text data given paths to train and development data files
    with open(train_file, 'r') as f1:
        sentences = f1.read().split('.')
    vocab1 = set(" ".join(sentences).split())

    word2id = {w: i for i, w in enumerate(list(vocab1))}
    s = map(lambda x: x.split(), sentences)


    data = []
    l=[]
    for sentence in s:
        for word_index in range(len(sentence)):
            l.append(sentence[word_index])

    l = np.array(l)
    a = l.shape[0]-2
    for word_index in range(a):
        t=[]
        for firword in l[word_index:word_index+3]:
            t.append(word2id[firword])
        data.append(t)
    data = np.array(data)
    train_sz = len(word2id)

    with open(dev_file, 'r') as f2:
        sentences2 = f2.read().split('.')
    s1 = map(lambda x: x.split(), sentences2)

    datadev = []
    l1 = []
    for sentence in s1:
        for word_index in range(len(sentence)):
            l1.append(sentence[word_index])

    l1 = np.array(l1)
    for word_index in range(l1.shape[0] - 2):
        t1 = []
        for firword in l1[word_index:word_index + 3]:
            t1.append(word2id[firword])
        datadev.append(t1)
    datadev = np.array(datadev)
    return train_sz, data, datadev


def main():

    # TODO: Import and process data
    vocabsz, data, datadev = read('train.txt', 'dev.txt')
    batchsz = 20

    # TODO: Set up placeholders for inputs and outputs to pass into model's constructor
    x1_tensor = tf.placeholder(dtype=tf.int32, shape=[None])
    x2_tensor = tf.placeholder(dtype=tf.int32, shape=[None])
    y_tensor = tf.placeholder(dtype=tf.int32, shape=[None])
    # TODO: Initialize model and tensorflow variables
    model = TrigramLM(x1_tensor, x2_tensor, y_tensor, vocabsz)
    sess = tf.Session()
    loss = model.loss
    train_op = model.train_op
    sess.run(tf.global_variables_initializer())
    # TODO: Set up the training step, training with 1 epoch and with a batch size of 20

    for start, end in zip(range(0, len(data) - batchsz, batchsz), range(batchsz, len(data), batchsz)):
        sess.run([train_op], feed_dict={model.X1: data[start:end, 0], model.X2: data[start:end, 1],
                                        model.Y: data[start:end, 2]})

    # TODO: Run the model on the development set and print the final perplexity
    # Remember that perplexity is just defined as: e^(average_loss_per_input)!
    avgloss = sess.run(loss, feed_dict={model.X1: datadev[:, 0], model.X2: datadev[:, 1],
                                      model.Y: datadev[:, 2]})

    print("The perplexity is:")
    print(np.exp(avgloss))


if __name__ == "__main__":
    main()

