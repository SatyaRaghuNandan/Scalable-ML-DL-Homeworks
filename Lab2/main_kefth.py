import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--relu', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--dp', type=float, default=0.85)
parser.add_argument('--proc', type=int, default=4)
args = parser.parse_args()

# load data
mnist = input_data.read_data_sets('data', one_hot=True)

# Model parameters

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

_HEIGHT = 28
_WIDTH = 28

# 1. Define Variables and Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# define relu parameter using a placeholder
act_function = tf.placeholder(tf.bool)
is_training = tf.placeholder(tf.bool)

# placeholders for both dropout and learning rate
lr_decay = tf.placeholder(tf.float32)
dropout = tf.placeholder(tf.float32)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.random_normal(shape, 0.1, stddev=0.05)
	return tf.Variable(initial)

def conv2d(x, W, strides=[1,1,1,1]):
	return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_norm(inputs, is_training):
	return tf.layers.batch_normalization(
      inputs=inputs, axis=3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)

"""

Performances: 

3 conv + 3 pool + 1 fc (0.75 dp) = 0.85 
3 conv + 3 pool + 1 fc (0.85 dp) = 0.86
3 conv + 3 pool + 1 fc (1 dp) = 0.86
3 conv + 3 pool + 2 fc (0.85 dp, 0.1 lr) = 0.86

"""

# augmentation
def data_preparation(image_batch):
	x_image = tf.reshape(image_batch, [-1, 28, 28, 1])
	print(x_image.get_shape())
	x_image_normalized = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x_image)
	x_image_flipped_left_right = tf.map_fn(lambda frame: tf.image.random_flip_left_right(frame), x_image_normalized)
	x_image_flipped_up_down = tf.map_fn(lambda frame: tf.image.random_flip_up_down(frame), x_image_flipped_left_right)
	batch_X, batch_Y = mnist.train.next_batch(100)
	train_dict = {x: batch_X, y_: batch_Y, lr_decay: args.lr, dropout: args.dp, act_function: args.relu, is_training: True}
	sess = tf.Session()
	reshaped = sess.run(tf.reshape(x_image_flipped_up_down[0], [28,28]), feed_dict=train_dict)
	x_image = sess.run(tf.reshape(x_image_flipped_up_down[0], [28,28]), feed_dict=train_dict)
	#reshaped = np.zeros(shape=(28,28))
	plt.imshow(reshaped, cmap="Greys")
	plt.show()
	plt.imshow(x_image, cmap="Greys")
	plt.show()
	return x_image_flipped_up_down

# building the model
def model(x, relu, prob, is_training):

	# initialize all the convolution variables
	W_conv1 = weight_variable([3, 3, 1, 32])
	b_conv1 = bias_variable([32])

	W_conv2 = weight_variable([3, 3, 32, 64])
	b_conv2 = bias_variable([64])

	W_conv3 = weight_variable([3, 3, 64, 128])
	b_conv3 = bias_variable([128])

	W1 = weight_variable([7*7*128,200])
	b1 = bias_variable([200])

	W2 = weight_variable([200,128])
	b2 = bias_variable([128])
	
	Wout = weight_variable([128,10])
	bout = bias_variable([10])

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	h_conv1 = conv2d(x_image, W_conv1, strides=[1,1,1,1]) + b_conv1
	h_conv1 = tf.nn.relu(h_conv1)
	h_conv1 = max_pool_2x2(h_conv1)

	h_conv2 = conv2d(h_conv1, W_conv2, strides=[1,1,1,1]) + b_conv2
	h_conv2 = tf.nn.relu(h_conv2)
	h_conv2 = max_pool_2x2(h_conv2)
	
	h_conv3 = conv2d(h_conv2, W_conv3, strides=[1,1,1,1]) + b_conv3
	h_conv3 = tf.nn.relu(h_conv3)

	x_flat = tf.reshape(h_conv3, [-1, 7*7*128])

	fc1 = tf.nn.relu(tf.matmul(x_flat, W1) + b1)
	fc1 = tf.nn.dropout(fc1, prob)

	fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
	fc2 = tf.nn.dropout(fc2, prob)

	y_logits = tf.matmul(fc1, Wout) + bout
	output = tf.nn.softmax(y_logits) 

	return output, y_logits

# res net
def res_net_model(x, relu, prob, is_training):

	# initialize all the convolution variables
	W_conv1 = weight_variable([3, 3, 1, 64])
	b_conv1 = bias_variable([64])
	W_conv2 = weight_variable([3, 3, 64, 64])
	b_conv2 = bias_variable([64])
	W_conv3 = weight_variable([3, 3, 64, 64])
	b_conv3 = bias_variable([64])

	W_res1 = weight_variable([3, 3, 64, 64])
	b_res1 = bias_variable([64])
	W_res2 = weight_variable([3, 3, 64, 64])
	b_res2 = bias_variable([64])
	W_res3 = weight_variable([3, 3, 64, 64])
	b_res3 = bias_variable([64])

	W_res_out = weight_variable([3, 3, 64, 10])
	b_res_out = bias_variable([10])

	W1 = weight_variable([7*7*128,200])
	b1 = bias_variable([200])

	W2 = weight_variable([200,128])
	b2 = bias_variable([128])
	
	Wout = weight_variable([128,10])
	bout = bias_variable([10])

	#x_image = tf.reshape(x, [-1, 28, 28, 1])

	# three conv layers (last with stride 2 to reduce dimension)
	h_conv1 = conv2d(x_image, W_conv1, strides=[1,1,1,1]) + b_conv1
	h_conv1 = tf.nn.relu(h_conv1)
	
	h_conv2 = conv2d(h_conv1, W_conv2, strides=[1,1,1,1]) + b_conv2
	h_conv2 = tf.nn.relu(h_conv2)

	h_conv3 = conv2d(h_conv2, W_conv3, strides=[1,2,2,1]) + b_conv3
	h_conv3 = tf.nn.relu(h_conv3)

	# residual learning 
	
	# batch normalization
	h_conv3 = batch_norm(h_conv3, is_training)

	h_res1 = conv2d(h_conv3, W_res1, strides=[1,1,1,1]) + b_res1
	h_res1 = tf.nn.relu(h_res1)
	
	# batch normalization
	h_res1 = batch_norm(h_res1, is_training)

	h_res2 = conv2d(h_res1, W_res2, strides=[1,1,1,1]) + b_res2
	h_res2 = tf.nn.relu(h_res2)

	# merging of the layers
	h_res_inp = tf.add(h_res2, h_conv3)

	h_res3 = conv2d(h_res_inp, W_res3, strides=[1,2,2,1]) + b_res3
	h_res3 = tf.nn.relu(h_conv3)

	h_out_conv = conv2d(h_res3, W_res_out, strides=[1,1,1,1]) + b_res_out

	pool = tf.layers.average_pooling2d(h_out_conv, 14, 14, padding='SAME')

	y_logits = tf.reshape(pool, [-1, 10])
	output = tf.nn.softmax(y_logits) 

	return output, y_logits

x_ = data_preparation(x)
prediction, y = res_net_model(x_, args.relu, dropout, is_training)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
global_step = tf.Variable(0, trainable=False)

k = 0.5
decay_rate = 0.96
learning_rate = tf.train.inverse_time_decay(lr_decay, global_step, k, decay_rate)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

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
	print(batch_X.sum())
    # the backpropagation training step
	train_dict = {x: batch_X, y_: batch_Y, lr_decay: args.lr, dropout: args.dp, act_function: args.relu, is_training: True}
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
		test_dict = {x: mnist.test.images, y_: mnist.test.labels, lr_decay: args.lr, dropout: 1.0, act_function: args.relu, is_training: False}
		a, c = sess.run([accuracy, cross_entropy], feed_dict=test_dict)
		test_a.append(a)
		test_c.append(c)
		print(a)
	return (train_a, train_c, test_a, test_c)

train_a = []
train_c = []
test_a = []
test_c = []
    
training_iter = 2000
epoch_size = 100
for i in range(training_iter):
    test = False
        test = True
    a, c, ta, tc = training_step(i, test, test)
    train_a += a
    train_c += c
    if i % epoch_size == 0:
    test_a += ta
    test_c += tc

if args.relu == True:
	print("Using RELU")
else:
	print("Using SIGMOID")

test_dictionary = {x: mnist.test.images, y_: mnist.test.labels, lr_decay: args.lr, dropout: 1.0, act_function: args.relu, is_training: False}
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
"""zoom_point = 50
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
plt.show()"""