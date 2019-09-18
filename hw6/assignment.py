import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_FR = "./processed_data/french_train.txt"
TRAIN_EN = "./processed_data/english_train.txt"
TEST_FR = "./processed_data/french_test.txt"
TEST_EN = "./processed_data/english_test.txt"

# This variable is the batch size the auto-grader will use when training your model.
BATCH_SIZE = 100

# Do not change these variables.
FRENCH_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 12
STOP_TOKEN = "*STOP*"


def pad_corpus(french_file_name, english_file_name):
    """
    Arguments are files of French, English sentences. All sentences are padded with "*STOP*" at
    the end to make their lengths match the window size. For English, an additional "*STOP*" is
    added to the beginning. For example, "I am hungry ." becomes
    ["*STOP*, "I", "am", "hungry", ".", "*STOP*", "*STOP*", "*STOP*",  "*STOP", "*STOP", "*STOP", "*STOP", "*STOP"]

    :param french_file_name: string, a path to a french file
    :param english_file_name: string, a path to an english file

    :return: A tuple of: (list of padded sentences for French, list of padded sentences for English, list of French sentence lengths, list of English sentence lengths)
    """

    french_padded_sentences = []
    french_sentence_lengths = []
    with open(french_file_name, 'rt', encoding='latin') as french_file:
        for line in french_file:
            padded_french = line.split()[:FRENCH_WINDOW_SIZE]
            french_sentence_lengths.append(len(padded_french))
            padded_french += [STOP_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_french))
            french_padded_sentences.append(padded_french)

    english_padded_sentences = []
    english_sentence_lengths = []
    with open(english_file_name, "rt", encoding="latin") as english_file:
        for line in english_file:
            padded_english = line.split()[:ENGLISH_WINDOW_SIZE]
            english_sentence_lengths.append(len(padded_english))
            padded_english = [STOP_TOKEN] + padded_english + [STOP_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_english))
            english_padded_sentences.append(padded_english)

    return french_padded_sentences, english_padded_sentences, french_sentence_lengths, english_sentence_lengths


class Model:
    """
        This is a seq2seq model.

        REMINDER:

        This model class provides the data structures for your NN,
        and has functions to test the three main aspects of the training.
        The data structures should not change after each step of training.
        You can add to the class, but do not change the
        function headers or return types.
        Make sure that these functions work with a loop to call them multiple times,
        instead of implementing training over multiple steps in the function
    """

    def __init__(self, french_window_size, english_window_size, french_vocab_size, english_vocab_size):
        """
        Initialize a Seq2Seq Model with the given data.

        :param french_window_size: max len of French padded sentence, integer
        :param english_window_size: max len of English padded sentence, integer
        :param french_vocab_size: Vocab size of french, integer
        :param english_vocab_size: Vocab size of english, integer
        """

        # Initialize Placeholders
        self.french_vocab_size, self.english_vocab_size = french_vocab_size, english_vocab_size

        self.encoder_input = tf.placeholder(tf.int32, shape=[None, french_window_size], name='french_input')
        self.encoder_input_length = tf.placeholder(tf.int32, shape=[None], name='french_length')

        self.decoder_input = tf.placeholder(tf.int32, shape=[None, english_window_size], name='english_input')
        self.decoder_input_length = tf.placeholder(tf.int32, shape=[None], name='english_length')
        self.decoder_labels = tf.placeholder(tf.int32, shape=[None, english_window_size], name='english_labels')

        # Please leave these variables
        self.logits = self.forward_pass()
        self.loss = self.loss_function()
        self.train = self.back_propagation()
        self.accuracy = self.accuracy_function()

    def forward_pass(self):
        """
        Calculates the logits

        :return: A tensor of size [batch_size, english_window_size, english_vocab_size]
        """
        EMBEDDINGSZ = 100
        rnnSz = 250
        keepprob = 0.5
        #print(self.encoder_input.shape)
        with tf.variable_scope("enc"):
            F = tf.Variable(tf.random_normal((self.french_vocab_size, EMBEDDINGSZ), stddev=.1))
            embs = tf.nn.embedding_lookup(F, self.encoder_input)
            embs = tf.nn.dropout(embs, keepprob)
            embs = tf.reshape(embs, [-1, FRENCH_WINDOW_SIZE, EMBEDDINGSZ])
            cell = tf.contrib.rnn.GRUCell(rnnSz)
            #initState = cell.zero_state(BATCH_SIZE, tf.float32)
            encOut, encState = tf.nn.dynamic_rnn(cell, embs, dtype=tf.float32)

            wad = tf.Variable(tf.truncated_normal([FRENCH_WINDOW_SIZE, ENGLISH_WINDOW_SIZE], stddev=.1))
            encOT = tf.transpose(encOut, [0, 2, 1])
            decIT = tf.tensordot(encOT, wad, [[2], [0]])
            decI = tf.transpose(decIT, [0, 2, 1])

        with tf.variable_scope("dec"):
            E = tf.Variable(tf.random_normal((self.english_vocab_size, EMBEDDINGSZ), stddev=.1))
            embs1 = tf.nn.embedding_lookup(E, self.decoder_input)
            embs1 = tf.nn.dropout(embs1, keepprob)
            embs1 = tf.reshape(embs1, [-1, ENGLISH_WINDOW_SIZE, EMBEDDINGSZ])
            embs2 = tf.concat([decI, embs1], 2)
            cell2 = tf.contrib.rnn.GRUCell(rnnSz)
            decOut, _ = tf.nn.dynamic_rnn(cell2, embs2, initial_state=encState)
            W = tf.Variable(tf.random_normal([rnnSz, self.french_vocab_size], stddev=.1))
            b = tf.Variable(tf.random_normal([self.french_vocab_size], stddev=.1))
            logits = tf.tensordot(decOut, W, axes=[[2], [0]]) + b
            return logits

    def loss_function(self):
        """
        Calculates the model cross-entropy loss after one forward pass

        :return: the loss of the model as a tensor (averaged over batch)
        """
        weights = tf.sequence_mask(self.decoder_input_length, ENGLISH_WINDOW_SIZE, dtype=tf.float32)
        return tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.decoder_labels, weights=weights)

    def back_propagation(self):
        """
        Adds optimizer to computation graph

        :return: optimizer
        """
        return tf.train.AdamOptimizer(1e-3).minimize(tf.reduce_sum(self.loss))

    def accuracy_function(self):

        labels = tf.sequence_mask(self.decoder_input_length, ENGLISH_WINDOW_SIZE, dtype=tf.int32)
        logits = tf.sequence_mask(self.decoder_input_length, ENGLISH_WINDOW_SIZE, dtype=tf.int64)
        return tf.metrics.accuracy(labels=tf.multiply(self.decoder_labels, labels),
                                   predictions=tf.multiply(tf.argmax(self.logits, 2), logits))


def main():
    # Load padded corpus
    train_french, train_english, train_french_lengths, train_english_lengths = pad_corpus(TRAIN_FR, TRAIN_EN)
    test_french, test_english, test_french_lengths, test_english_lengths = pad_corpus(TEST_FR, TEST_EN)

    # 1: Build French, English Vocabularies (dictionaries mapping word types to int ids)
    with open(TRAIN_FR, 'rt', encoding='latin') as french_file:
        sentences = french_file.read().split(' ')
        sentences += [STOP_TOKEN]
    vocab1 = set(" ".join(sentences).split())
    word2id1 = {w: i for i, w in enumerate(list(vocab1))}
    with open(TRAIN_EN, 'rt', encoding='latin') as english_file:
        sentences2 = english_file.read().split(' ')
        sentences2 += [STOP_TOKEN]
    vocab2 = set(" ".join(sentences2).split())
    word2id2 = {w: i for i, w in enumerate(list(vocab2))}
    fvocabsize = len(word2id1)
    evocabsize = len(word2id2)

    train_english = np.array(train_english)
    train_french = np.array(train_french)
    fps = np.zeros((len(train_french), train_french.shape[1]))
    eps = np.zeros((len(train_english), train_english.shape[1]))
    #print("a")
    for i in range(len(train_french)):
        for j in range(train_french.shape[1]):
            fps[i][j] = word2id1[train_french[i][j]]
    #print("b")
    for i in range(len(train_english)):
        for j in range(train_english.shape[1]):
            eps[i][j] = word2id2[train_english[i][j]]
    input = eps[:, 0: ENGLISH_WINDOW_SIZE]
    label = eps[:, 1: ENGLISH_WINDOW_SIZE+1]

    test_english = np.array(test_english)
    test_french = np.array(test_french)
    ft = np.zeros((len(test_french), test_french.shape[1]))
    et = np.zeros((len(test_english), test_english.shape[1]))
    for i in range(len(test_french)):
        for j in range(test_french.shape[1]):
            ft[i][j] = word2id1[test_french[i][j]]
    #print("b")
    for i in range(len(test_english)):
        for j in range(test_english.shape[1]):
            et[i][j] = word2id2[test_english[i][j]]
    testinput = et[:, 0: ENGLISH_WINDOW_SIZE]
    testlabel = et[:, 1: ENGLISH_WINDOW_SIZE+1]

    # 2: Creates batches. Remember that the English Decoder labels need to be shifted over by 1.
    # 3. Initialize model
    model = Model(FRENCH_WINDOW_SIZE, ENGLISH_WINDOW_SIZE, fvocabsize, evocabsize)

    loss = model.loss
    train_op = model.train
    accuracy = model.accuracy
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # 4: Launch Tensorflow Session
    #       -Train
    lossc = 0
    step = 0
    for start, end in zip(range(0, len(train_french_lengths) - BATCH_SIZE, BATCH_SIZE),
                          range(BATCH_SIZE, len(train_french_lengths), BATCH_SIZE)):
        l, _ = sess.run([loss, train_op], feed_dict={model.encoder_input: fps[start:end],
                                                     model.encoder_input_length: train_french_lengths[start:end],
                                                     model.decoder_input: input[start:end],
                                                     model.decoder_input_length: train_english_lengths[start:end],
                                                     model.decoder_labels: label[start:end]})
        step = step + 1
        lossc += l
        if start % 10000 == 0:
            print(' 10K Training data in %d\t  Loss: %.3f ' % (start // 10000, lossc / step))
    #       -Test
    sum = 0
    step = 0
    ll = 0.0
    for start, end in zip(range(0, len(test_french_lengths) - BATCH_SIZE, BATCH_SIZE),
                          range(BATCH_SIZE, len(test_french_lengths), BATCH_SIZE)):
        if len(test_french_lengths[start:end]) != BATCH_SIZE:
            break
        l, acc = sess.run([loss, accuracy], feed_dict={model.encoder_input: ft[start:end],
                                                       model.encoder_input_length: test_french_lengths[start:end],
                                                       model.decoder_input: testinput[start:end],
                                                       model.decoder_input_length: test_english_lengths[start:end],
                                                       model.decoder_labels: testlabel[start:end]})
        step = step + 1
        ll += l
        sum += acc[0]
        if start % 1000 == 0:
            print(' 1K Testing data in %d\t  Loss: %.3f perplexity: %.3f accuracy:  %.3f' % (start // 1000, ll / step,
                                                                                             np.exp(ll / step), sum / step))


if __name__ == '__main__':
    main()



"""

Human: What do we want!?
Computer: Natural language processing!
Human: When do we want it!?
Computer: When do we want what?

"""
