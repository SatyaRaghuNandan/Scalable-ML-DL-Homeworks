import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--proc', type=int, default=4)
parser.add_argument('--activation', type=str, default="sigmoid")
parser.add_argument('--optimizer', type=str, default="gd")
parser.add_argument('--dropout', type=float, default=0.75)
parser.add_argument('--decay', type=str, default="itd")
args = parser.parse_args()

# load data
mnist = input_data.read_data_sets('data/fashion', one_hot=True)

# 1. Define Variables and Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# define relu parameter suing a placeholder
act_function = tf.placeholder(tf.bool)

# placeholders for both dropout and learning rate
lr_decay = tf.placeholder(tf.float32)
dropout = tf.placeholder(tf.float32)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, strides=[1,1,1,1]):
	return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# building the model
def model(x, prob):

	# initialize all the convolution variables
	W_conv1 = weight_variable([5, 5, 1, 4])
	b_conv1 = bias_variable([4])

	W_conv2 = weight_variable([5, 5, 4, 8])
	b_conv2 = bias_variable([8])

	W_conv3 = weight_variable([4, 4, 8, 12])
	b_conv3 = bias_variable([12])

	# fully connected layers variables
	W1 = weight_variable([7*7*12,200])
	b1 = bias_variable([200])
	W2 = weight_variable([200,10])
	b2 = bias_variable([10])

	# reshaping the input (from vector to matrix)
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	def activation_function(in_):
		if args.activation == "relu":
			return tf.nn.relu(in_)
		elif args.activation == "sigmoid":
			return tf.nn.sigmoid(in_)

	h_conv1 = activation_function(conv2d(x_image, W_conv1, strides=[1,1,1,1]) + b_conv1)
	print(h_conv1.get_shape())
	h_conv2 = activation_function(conv2d(h_conv1, W_conv2, strides=[1,2,2,1]) + b_conv2)
	print(h_conv2.get_shape())
	h_conv3 = activation_function(conv2d(h_conv2, W_conv3, strides=[1,2,2,1]) + b_conv3)
	print(h_conv3.get_shape())

	x_flat = tf.reshape(h_conv3, [-1, 7*7*12])

	y1 = activation_function(tf.matmul(x_flat, W1) + b1)
	y1_dropout = tf.nn.dropout(y1, prob)
	y_logits = tf.matmul(y1_dropout, W2) + b2
	output = tf.nn.softmax(y_logits) 

	return output, y_logits

prediction, y = model(x, dropout)
# 3. Define the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 4. Define the accuracy
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 5. Define an optimizer
global_step = tf.Variable(0, trainable=False)
k = 0.5
decay_rate = 0.96
if args.optimizer == "adam":
	if args.decay == "itd":
		learning_rate = tf.train.inverse_time_decay(lr_decay, global_step, k, decay_rate)
	else:
		learning_rate = 0.5
	optimizer = tf.train.AdamOptimizer(learning_rate) #0.8784
elif args.optimizer == "gd":
	if args.decay == "itd":
		learning_rate = tf.train.inverse_time_decay(lr_decay, global_step, k, decay_rate)
	else:
		learning_rate = 0.5
	optimizer = tf.train.GradientDescentOptimizer(learning_rate) #0.859

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
	train_dict = {x: batch_X, y_: batch_Y, lr_decay: 0.0001, dropout: 0.75}
	sess.run(train_step, feed_dict=train_dict)
    
    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
	train_a = []
	train_c = []
	test_a = []
	test_c = []

	test_dict = {x: mnist.test.images, y_: mnist.test.labels, lr_decay: 0.0001, dropout: 1.0}

	if update_train_data:
		a, c = sess.run([accuracy, cross_entropy], feed_dict=train_dict)
		train_a.append(a)
		train_c.append(c)

	if update_test_data:
		a, c = sess.run([accuracy, cross_entropy], feed_dict=test_dict)
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

test_dictionary = {x: mnist.test.images, y_: mnist.test.labels, lr_decay: 0.1, dropout: 1.0}
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
1. Setup the network layer with 3 conv layers, 1 relu layer and 1 softmax layer with a GradientDescentOptimizer.
QUESTION 1:

QUESTION 2:

QUESTION 3:

QUESTION 4:
0.8932 -> sigmoid [lr 0.8] GD
0.8974 -> relu [lr 0.1] GD
0.8783 -> relu [lr 0.0001] ADAM

"""