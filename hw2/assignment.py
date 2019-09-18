import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE,
# YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.
class Model:

    def __init__(self, image, label):
        """
        A Model class contains a computational graph that classifies images
        to predictions. Each of its methods builds part of the graph
        on Model initialization.

        image: the input image to the computational graph as a tensor
        label: the correct label of an image as a tensor
        prediction: the output prediction of the computational graph,
                    produced by self.forward_pass()
        optimize: the model's optimizing tensor produced by self.optimizer()
        loss: the model's loss produced by computing self.loss_function()
        accuracy: the model's prediction accuracy – no need to modify this
        """
        self.image = image
        self.label = label
        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()

    def forward_pass(self):
        """
        Predicts a label given an image using fully connected layers

        :return: the predicted label as a tensor
        """
        # TODO replace pass with forward_pass method
        W = tf.Variable(tf.random_normal([784, 10], stddev=.1))
        b = tf.Variable(tf.random_normal([10], stddev=.1))
        U = tf.Variable(tf.random_normal([784, 784], stddev=.1))
        bU = tf.Variable(tf.random_normal([784], stddev=.1))
        layer = tf.matmul(self.image, U) + bU
        layer = tf.nn.relu(layer)
        print("aa")
        return tf.matmul(layer, W) + b


    def loss_function(self):
        """
        Calculates the model loss

        :return: the loss of the model as a tensor
        """
        # TODO replace pass with loss_function method
        return tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.prediction)

    def optimizer(self):
        """
        Optimizes the model loss

        :return: the optimizer as a tensor
        """
        # TODO replace pass with optimizer method
        learning_rate = 0.5
        sgd = tf.train.GradientDescentOptimizer(learning_rate)
        return sgd.minimize(self.loss_function())

    def accuracy_function(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():

    # TODO: import MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # TODO: Set-up placeholders for inputs and outputs
    batchScz = 100
    x_tensor = tf.placeholder(dtype=tf.float32, shape=[batchScz, 784])
    y_tensor = tf.placeholder(dtype=tf.float32, shape=[batchScz, 10])

    # TODO: initialize model and tensorflow variables

    model = Model(x_tensor, y_tensor)
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    loss = model.loss_function()
    train_op = model.optimizer()

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
    print("Test Accuracy: %r" % (sum/100))
    return


if __name__ == '__main__':
    main()
