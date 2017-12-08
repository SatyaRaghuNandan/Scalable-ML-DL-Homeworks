# all tensorflow api is accessible through this
import tensorflow as tf        
# to visualize the resutls
import matplotlib.pyplot as plt 
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data


alpha = 0.003
layer_dim = [200,100,60,30,10]
keeping_probability = 0.85
tf.set_random_seed(0)

# load data
mnist = input_data.read_data_sets('data', one_hot=True)

# 1. Define Variables and Placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.random_normal(shape, mean=0.5, stddev=0.3)
  return tf.Variable(initial)

# 2. Define the model
y1_conv = tf.nn.conv2d(tf.reshape(x, [-1, 28, 28, 1]), weight_variable([5,5,1,4]), strides=[1,1,1,1], padding="SAME") + bias_variable([4])
y2_conv = tf.nn.conv2d(y1_conv, weight_variable([5,5,4,8]), strides=[1,2,2,1], padding="SAME") + bias_variable([8])
y3_conv = tf.nn.conv2d(y2_conv, weight_variable([4,4,8,12]), strides=[1,2,2,1], padding="SAME") + bias_variable([12])
y3_flat = tf.reshape(y3_conv, [-1, 7*7*12])
y3_logits = tf.matmul(y3_flat, weight_variable([7*7*12, 200])) + bias_variable([200])
densely_connected = tf.nn.relu(y3_logits)
y_readout_logit = tf.matmul(densely_connected, weight_variable([200, 10])) + bias_variable([10])
y_readout = tf.nn.softmax(y_readout_logit)

# 3. Define the loss function  
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_readout_logit))

# 4. Define the accuracy 
is_correct = tf.equal(tf.argmax(y_readout, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Defininf the decaying learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(alpha, global_step, 100, 0.96, staircase=False)

# 5. Define an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# initialize
init = tf.initialize_all_variables()
config = tf.ConfigProto(intra_op_parallelism_threads=4)
sess = tf.Session(config=config)
sess.run(init)


def training_step(i, update_test_data, update_train_data):

    if i%1000 == 0:
        print(i)

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


# 6. Train and test the model, store the accuracy and loss per iteration

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
    
# 7. Plot and visualise the accuracy and loss

# accuracy training vs testing dataset
plt.title("Accuracy")
plt.plot(train_a, label="train")
plt.plot(test_a, label="test")
plt.grid(True)
plt.show()

# loss training vs testing dataset
plt.title("Loss")
plt.plot(train_c, label="train")
plt.plot(test_c, label="test")
plt.grid(True)
plt.show()

# Zoom in on the tail of the plots
plt.title("Accuracy zoom")
zoom_point = 50
x_range = range(zoom_point,training_iter/epoch_size)
plt.plot(x_range, train_a[zoom_point:], label="train")
plt.plot(x_range, test_a[zoom_point:], label="test")
plt.grid(True)
plt.show()

plt.title("Loss zoom")
plt.plot(train_c[zoom_point:], label="train")
plt.plot(test_c[zoom_point:], label="test")
plt.grid(True)
plt.show()
    
