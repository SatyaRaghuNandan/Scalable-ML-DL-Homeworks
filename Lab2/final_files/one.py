import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--proc', type=int, default=4)
parser.add_argument('--optimizer', type=str, default="gd")
args = parser.parse_args()

# load data
mnist = input_data.read_data_sets('data/fashion', one_hot=True)

# 1. Define Variables and Placeholders
# the last one rapresents the input channels (grey --> 1 channel, RGB --> 3 channels)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.truncated_normal([10], stddev=0.1))
# 2. Define the model (output of the model)
y = tf.nn.softmax(tf.matmul(x, W) + b) 
# 3. Define the loss function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 4. Define the accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 5. Define an optimizer
if args.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(0.0009)
elif args.optimizer == "gd":
    optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)
# initialize
init = tf.initialize_all_variables() 
config = tf.ConfigProto(intra_op_parallelism_threads=args.proc, inter_op_parallelism_threads=args.proc)
sess = tf.Session(config=config)
sess.run(init)

def training_step(i, update_test_data, update_train_data):

    if i % 100 == 0:
    	print("\r", i)
    ####### actual learning 
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={x: batch_X, y_: batch_Y})
    
    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={x: batch_X, y_: batch_Y})
        train_a.append(a)
        train_c.append(c)

    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        test_a.append(a)
        test_c.append(c)
    return (train_a, train_c, test_a, test_c)

train_a = []
train_c = []
test_a = []
test_c = []
    
training_iter = 10000
epoch_size = 100
for i in range(training_iter):
    test = False
    if i % epoch_size == 0:
        test = True
    a, c, ta, tc = training_step(i, test, test)
    train_a += a
    train_c += c
    test_a += ta
    test_c += tc

test_dictionary = {x: mnist.test.images, y_: mnist.test.labels}
print("Test accuracy: ", sess.run(accuracy, feed_dict=test_dictionary))
    
# accuracy training vs testing dataset
plt.title("Accuracy")
plt.plot(train_a, label="train")
plt.plot(test_a, label="test")
plt.grid(True)
plt.legend(loc="best")
plt.show()

# loss training vs testing dataset
plt.title("Loss")
plt.plot(train_c, label="train")
plt.plot(test_c, label="test")
plt.grid(True)
plt.legend(loc="best")
plt.show()

# Zoom in on the tail of the plots
zoom_point = 50
x_range = range(zoom_point,int(training_iter/epoch_size))
plt.title("Accuracy Zoom")
plt.plot(x_range, train_a[zoom_point:], label="train")
plt.plot(x_range, test_a[zoom_point:], label="test")
plt.grid(True)
plt.legend(loc="best")
plt.show()

plt.title("Loss Zoom")
plt.plot(train_c[zoom_point:], label="train")
plt.plot(test_c[zoom_point:], label="test")
plt.grid(True)
plt.legend(loc="best")
plt.show()


"""
QUESTION 1:
With Gradient descent and learning rate of 0.003 after 10000 iterations both the accuracy and the loss reamins more or 
less stable around the value respectively of 83% and 6000.
With Adam optimizer (and learning rate 0.0009) we get to 84% of accuracy and 4500 points of loss.

With the provided parameters for the learning rates both optimizers diverge.

The model seems to be under fitting because the loss never goes down a certain threshold.
"""












