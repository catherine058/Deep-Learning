import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:

    def __init__(self, image, label):
        """
        A Model class contains a computational graph that classifies images
        to predictions. Each of its methods builds part of the graph
        on Model initialization. Do not modify the constructor, as doing so
        would break the autograder. You may, however, add class variables
        to use in your graph-building. e.g. learning rate, 

        image: the input image to the computational graph as a tensor
        label: the correct label of an image as a tensor
        prediction: the output prediction of the computational graph,
                    produced by self.forward_pass()
        optimize: the model's optimizing tensor produced by self.optimizer()
        loss: the model's loss produced by computing self.loss_function()
        accuracy: the model's prediction accuracy
        """
        self.image = image
        self.label = label

        # TO-DO: Add any class variables you want to use.

        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()

    def forward_pass(self):
        """
        Predicts a label given an image using convolution layers

        :return: the prediction as a tensor
        """
        # TO-DO: Build up the computational graph for the forward pass.
        image = tf.reshape(self.image, [50, 28, 28, 1])
        flts = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        fb = tf.Variable(tf.truncated_normal([32], stddev=0.1))
        convOut = tf.nn.relu(tf.nn.conv2d(image, flts, [1, 1, 1, 1], "SAME")+fb)
        pool = tf.nn.max_pool(convOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        flts2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        fb2 = tf.Variable(tf.truncated_normal([64], stddev=0.1))
        convOut2 = tf.nn.relu(tf.nn.conv2d(pool, flts2, [1, 1, 1, 1], "SAME") + fb2)
        pool2 = tf.nn.max_pool(convOut2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool2 = tf.reshape(pool2, [50, 3136])
        W1 = tf.Variable(tf.truncated_normal([3136, 784], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal([784], stddev=0.1))
        pool3 = tf.matmul(pool2, W1) + b1
        W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([10], stddev=0.1))

        return tf.matmul(pool3, W) + b

    def loss_function(self):
        """
        Calculates the model cross-entropy loss

        :return: the loss of the model as a tensor
        """
        # TO-DO: Add the loss function to the computational graph
        return tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.prediction)

    def optimizer(self):
        """
        Optimizes the model loss using an Adam Optimizer

        :return: the optimizer as a tensor
        """
        # TO-DO: Add the optimizer to the computational graph
        sgd = tf.train.AdamOptimizer(1e-4)
        return sgd.minimize(self.loss)

    def accuracy_function(self):
        """
        Calculates the model's prediction accuracy by comparing
        predictions to correct labels – no need to modify this

        :return: the accuracy of the model as a tensor
        """
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():

    # TO-DO: import MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    batchScz = 50
    x_tensor = tf.placeholder(dtype=tf.float32, shape=[batchScz, 784])
    y_tensor = tf.placeholder(dtype=tf.float32, shape=[batchScz, 10])

    # TO-DO: Set-up placeholders for inputs and outputs
    model = Model(x_tensor, y_tensor)
    session = tf.Session()
    loss = model.loss_function()
    train_op = model.optimizer()
    session.run(tf.initialize_all_variables())

    # TODO: Set-up the training step, for as many of the 60,000 examples as you'd like
    #     where the batch size is greater than 1
    for i in range(2000):
        x_tensor, y_tensor = mnist.train.next_batch(batchScz)
        session.run(train_op, feed_dict={model.image: x_tensor, model.label: y_tensor})
    # TODO: run the model on test data and print the accuracy
    sum = 0
    accuracy = model.accuracy_function()
    for i in range(100):
        x_tensor, y_tensor = mnist.test.next_batch(batchScz)
        sum += session.run(accuracy, feed_dict={model.image: x_tensor, model.label: y_tensor})
    print("Test Accuracy: %r" % (sum / 100))

    # TO-DO: initialize model and tensorflow variables

    # TO-DO: Set-up the training step, for 2000 batches with a batch size of 50

    # TO-DO: run the model on test data and print the accuracy

    return


if __name__ == '__main__':
    main()
