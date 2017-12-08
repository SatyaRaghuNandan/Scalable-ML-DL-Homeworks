# all tensorflow api is accessible through this
import tensorflow as tf        
# to visualize the resutls
import matplotlib.pyplot as plt 
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data


alpha = 0.003
layer_dim = [200,100,60,30,10]

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

tf.set_random_seed(0)

# load data
mnist = input_data.read_data_sets('data', one_hot=True)

# 1. Define Variables and Placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.truncated_normal([784, layer_dim[0]], stddev=0.1)) 
b1 = tf.Variable(tf.random_normal([layer_dim[0]], mean=0.5, stddev=0.3))
w2 = tf.Variable(tf.truncated_normal([layer_dim[0], layer_dim[1]], stddev=0.1))
b2 = tf.Variable(tf.random_normal([layer_dim[1]], mean=0.5, stddev=0.3))
w3 = tf.Variable(tf.truncated_normal([layer_dim[1], layer_dim[2]], stddev=0.1))
b3 = tf.Variable(tf.random_normal([layer_dim[2]], mean=0.5, stddev=0.3))
w4 = tf.Variable(tf.truncated_normal([layer_dim[2], layer_dim[3]], stddev=0.1))
b4 = tf.Variable(tf.random_normal([layer_dim[3]], mean=0.5, stddev=0.3))
w5 = tf.Variable(tf.truncated_normal([layer_dim[3], layer_dim[4]], stddev=0.1))
b5 = tf.Variable(tf.zeros([layer_dim[4]]))

# 2. Define the model
y1_logits = tf.matmul(x, w1) + b1
y1 = lrelu(y1_logits, 0.001)
y2_logits = tf.matmul(y1, w2) + b2
y2 = lrelu(y2_logits, 0.001)
y3_logits = tf.matmul(y2, w3) + b3
y3 = lrelu(y3_logits, 0.001)
y4_logits = tf.matmul(y3, w4) + b4
y4 = lrelu(y4_logits, 0.001)
y5_logits = tf.matmul(y4, w5) + b5
y5 = tf.nn.softmax(y5_logits)

# 3. Define the loss function  
cross_entropy = -tf.reduce_sum(y_ * tf.log(y5))

# 4. Define the accuracy 
is_correct = tf.equal(tf.argmax(y5, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Defininf the decaying learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(alpha, global_step, 100, 0.96, staircase=False)
# 5. Define an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# initialize
init = tf.initialize_all_variables()
sess = tf.Session()
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
    
