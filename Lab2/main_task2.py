import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--activation', type=str, default="sigmoid")
parser.add_argument('--proc', type=int, default=4)
parser.add_argument('--optimizer', type=str, default="gd")
args = parser.parse_args()

# load data
mnist = input_data.read_data_sets('data/fashion', one_hot=True)
# 1. Define Variables and Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# building the model
def model(x):

	W1 = tf.Variable(tf.truncated_normal([784,200], stddev=0.1))
	b1 = tf.Variable(tf.zeros([200]))
	W2 = tf.Variable(tf.truncated_normal([200,100], stddev=0.1))
	b2 = tf.Variable(tf.zeros([100]))
	W3 = tf.Variable(tf.truncated_normal([100,60], stddev=0.1))
	b3 = tf.Variable(tf.zeros([60]))
	W4 = tf.Variable(tf.truncated_normal([60,30], stddev=0.1))
	b4 = tf.Variable(tf.zeros([30]))
	W5 = tf.Variable(tf.truncated_normal([30,10], stddev=0.1))
	b5 = tf.Variable(tf.zeros([10]))

	x_flat = tf.reshape(x, [-1, 784])

	def activation_function(in_):
		if args.activation == "relu":
			return tf.nn.relu(in_)
		elif args.activation == "sigmoid":
			return tf.nn.sigmoid(in_)

	y1 = activation_function(tf.matmul(x_flat, W1) + b1)
	y2 = activation_function(tf.matmul(y1, W2) + b2) 
	y3 = activation_function(tf.matmul(y2, W3) + b3) 
	y4 = activation_function(tf.matmul(y3, W4) + b4) 
	y_logits = tf.matmul(y4, W5) + b5
	output = tf.nn.softmax(y_logits) 

	return output, y_logits

prediction, y = model(x)

# 3. Define the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 4. Define the accuracy
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 5. Define an optimizer
if args.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(0.0009) #0.8784
elif args.optimizer == "gd":
    optimizer = tf.train.GradientDescentOptimizer(0.7) #0.859
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
	train_dict = {x: batch_X, y_: batch_Y}
	sess.run(train_step, feed_dict=train_dict)
    
    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
	train_a = []
	train_c = []
	test_a = []
	test_c = []
	if update_train_data:
		a, c = sess.run([accuracy, cross_entropy], feed_dict=train_dict)
		train_a.append(a)
		train_c.append(c)

	if update_test_data:
		test_dict = {x: mnist.test.images, y_: mnist.test.labels}
		a, c = sess.run([accuracy, cross_entropy], feed_dict= test_dict)
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

print("Using "+ args.activation.upper()+ " + " + args.optimizer.upper() + ".")

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
Using Relu + Adam test accuracy : 0.8673 on test set
Using Relu + gd test accuracy : 0.8563 on test set
Using Sigmoid + Adam test accuracy : 0.8784 on test set
Using Sigmoid + gd test accuracy : 0.86 on test set

QUESTION 2:
Relu is slightly faster in converging. The reason is that relu tends to reach convergence faster than saturating neurons.
Relu is also faster from the computational point of view not requiring any exponential computation, this also tends to fasten
the training (not the convergence).

QUESTION 3:
We use softmax because is the closest differentiable function to max that is thus trainable with gradient descent.

QUESTION 4:
With the final implementation we didn't notice any strange behaviour but in a previously built model we had the possibility
to observe the phenomenon of the dying relu. As a matter of fact due to excessively small gradient propagated through the 
network more and more relu units were dying and after interrupting a certain number of paths we were observing a sudden drop 
in the prediciton capabilities of the network.

"""