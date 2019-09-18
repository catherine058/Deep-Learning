"""
Stencil layout for your RNN language model assignment.
The stencil has three main parts:
    - A class stencil for your language model
    - A "read" helper function to isolate the logic of parsing the raw text files. This is where
    you should build your vocabulary, and transform your input files into data that can be fed into the model.
    - A main-training-block area - this code (under "if __name__==__main__") will be run when the script is,
    so it's where you should bring everything together and execute the actual training of your model.


Q: What did the computer call its father?
A: Data!

"""

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Use this variable to declare your batch size. Do not rename this variable.
BATCH_SIZE = 50

# Your window size must be 20. Do not change this variable!
WINDOW_SIZE = 20


def read(train_file, test_file):
    """
    Read and parse the file, building the vectorized representations of the input and output.

    !!!!!PLEASE FOLLOW THE STENCIL. WE WILL GRADE THIS!!!!!!!

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of (train_x, train_y, test_x, test_y, vocab)

    train_x: List of word ids to use as training input
    train_y: List of word ids to use as training labels
    test_x: List of word ids to use as testing input
    test_y: List of word ids to use as testing labels
    vocab: A dict mapping from word to vocab id
    """

    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also needo's to drop the first element from train_y and test_y.
    # If you don't do this, you will see very, very small perplexities.


    with open(train_file, 'r') as f1:
        sentences = f1.read().split('.')
    vocab1 = set(" ".join(sentences).split())
    word2id = {w: i for i, w in enumerate(list(vocab1))}
    s = map(lambda x: x.split(), sentences)
    l = []
    for sentence in s:
        for word in sentence:
            l.append(word2id[word])
    l = np.array(l)
    #print(l.shape)
    length = len(l)-1
    train_x = l[0:length]
    tnx = []
    for k in range(length//BATCH_SIZE//WINDOW_SIZE):
        for j in range(BATCH_SIZE):
            for i in range(WINDOW_SIZE):
                tnx.append(train_x[i+j*(len(train_x)//BATCH_SIZE)+k*WINDOW_SIZE])

    tnx = np.array(tnx)
    #print(tnx.shape)
    #train_x = np.reshape(train_x, [BATCH_SIZE, (length-1)/BATCH_SIZE])
    #print(train_x.shape)
    train_y = l[1:len(l)]
    tny = []
    for k in range(length // BATCH_SIZE // WINDOW_SIZE):
        for j in range(BATCH_SIZE):
            for i in range(WINDOW_SIZE):
                tny.append(train_y[i + j * (len(train_y) // BATCH_SIZE)+k*WINDOW_SIZE])
    #train_y = np.reshape(train_y, [BATCH_SIZE, (length-1) / BATCH_SIZE])
    #print(train_y.shape)
    tny = np.array(tny)

    with open(test_file, 'r') as f2:
        sentences2 = f2.read().split('.')
    s = map(lambda x: x.split(), sentences2)
    l2 = []
    for sentence in s:
        for word in sentence:
            l2.append(word2id[word])
    l2 = np.array(l2)
    length2 = len(l2)-1
    n = length2//WINDOW_SIZE
    test_x = l2[0:length2]
    ttx = l2[0:WINDOW_SIZE*n]
    #ttx = np.reshape(ttx, [length2//WINDOW_SIZE, WINDOW_SIZE])
    # train_y = np.reshape(train_y, [BATCH_SIZE, (length-1) / BATCH_SIZE])
    # print(train_y.shape)

    #test_x = np.reshape(test_x, [length2 / BATCH_SIZE, BATCH_SIZE])
    #print(test_x.shape)
    test_y = l2[1:len(l2)]
    tty = l2[1:WINDOW_SIZE*n+1]
    #tty = np.reshape(tty, [length2//WINDOW_SIZE, WINDOW_SIZE])
    # train_y = np.reshape(train_y, [BATCH_SIZE, (length-1) / BATCH_SIZE])
    # print(train_y.shape)
    #test_y = np.reshape(test_y, [length2 / BATCH_SIZE, BATCH_SIZE])
    #print(test_y.shape)

    return tnx, tny, ttx, tty, word2id


class Model:
    def __init__(self, inputs, labels, keep_prob, vocab_size):
        """
        The Model class contains the computation graph used to predict the next word in sequences of words.

        Do not delete any of these variables!

        inputs: A placeholder of input words
        label: A placeholder of next words
        keep_prob: The keep probability of dropout in the embeddings
        vocab_size: The number of unique words in the data
        """

        # Input tensors, DO NOT CHANGE
        self.inputs = inputs
        self.labels = labels
        self.keep_prob = keep_prob

        # DO NOT CHANGE
        self.vocab_size = vocab_size
        self.prediction = self.forward_pass()  # Logits for word predictions
        self.loss = self.loss_function()  # The average loss of the batch
        self.optimize = self.optimizer()  # An optimizer (e.g. ADAM)
        self.perplexity = self.perplexity_function()  # The perplexity of the model, Tensor of size 1

    def forward_pass(self):
        """
        Use self.inputs to predict self.labels.
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :return: logits: The prediction logits as a tensor
        """
        EMBEDDING_SIZE=100
        rnnSz=550
        E = tf.Variable(tf.random_normal([self.vocab_size, EMBEDDING_SIZE], stddev=0.1))
        embedding = tf.nn.embedding_lookup(E, self.inputs)
        #print(embedding.shape)
        #kp = self.keep_prob
        embedding = tf.reshape(embedding, [-1, WINDOW_SIZE, EMBEDDING_SIZE])
        embedding = tf.nn.dropout(embedding, self.keep_prob)
        rnn = tf.contrib.rnn.BasicLSTMCell(rnnSz)

        rnn = tf.nn.rnn_cell.DropoutWrapper(rnn, output_keep_prob=self.keep_prob)

        #initialState = rnn.zero_state(BATCH_SIZE, tf.float32)
        outputs, nextState = tf.nn.dynamic_rnn(rnn, embedding, dtype=tf.float32)
        #print(outputs.shape)
        W = tf.Variable(tf.truncated_normal([rnnSz, self.vocab_size],
                                            stddev=0.1))
        outputs = tf.reshape(outputs, [-1, rnnSz])
        #print(outputs.shape)
        logits = tf.matmul(outputs, W)
        #print(logits.shape)
        return logits

    def optimizer(self):
        """
        Optimizes the model loss using an Adam Optimizer
        :return: the optimizer as a tensor
        """
        return tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def loss_function(self):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :return: the loss of the model as a tensor of size 1
        """
        inputs = tf.reshape(self.inputs, [-1, WINDOW_SIZE])
        labels = tf.reshape(self.labels, [-1, WINDOW_SIZE])
        weights = tf.cast(tf.ones_like(inputs), dtype=tf.float32)
        logits = tf.reshape(self.prediction, [-1, WINDOW_SIZE, self.vocab_size])
        #print(logits.shape)
        return tf.contrib.seq2seq.sequence_loss(logits=logits, targets=labels, weights=weights)

    def perplexity_function(self):
        """
        Calculates the model's perplexity by comparing predictions to correct labels
        :return: the perplexity of the model as a tensor of size 1
        """
        return tf.exp(self.loss)



def main():
    # Preprocess data
    train_file = "train.txt"
    dev_file = "dev.txt"
    train_x, train_y, test_x, test_y, vocab_map = read(train_file, dev_file)
    vocab_size = len(vocab_map)
    # TODO: define placeholders
    x_tensor = tf.placeholder(dtype=tf.int32, shape=[None])
    y_tensor = tf.placeholder(dtype=tf.int32, shape=[None])
    kp = tf.placeholder(dtype=tf.float32)
    # TODO: initialize model
    model = Model(x_tensor, y_tensor, kp, vocab_size)
    sess = tf.Session()
    per = model.perplexity
    train_op = model.optimize
    sess.run(tf.global_variables_initializer())
    # TODO: Set-up the training step:
    step = 0
    for start, end in zip(range(0, len(train_x) - BATCH_SIZE*WINDOW_SIZE, BATCH_SIZE*WINDOW_SIZE),
                          range(BATCH_SIZE*WINDOW_SIZE, len(train_x), BATCH_SIZE*WINDOW_SIZE)):
        sess.run([train_op], feed_dict={model.inputs: train_x[start:end], model.labels: train_y[start:end],
                                        model.keep_prob: 0.5})
        step = step + 1
        if step % 50 == 0:
            print(step)

    per = sess.run(per, feed_dict={model.inputs: test_x, model.labels: test_y,
                                   model.keep_prob: 1.0})

    print("The perplexity is:")
    print(per)

    # - 1) divide training set into equally sized batch chunks. We recommend 50 batches.
    # - 2) split these batch segments into windows of size WINDOW_SIZE.

    # TODO: Run the model on the development set and print the final perplexity


if __name__ == '__main__':
    main()
