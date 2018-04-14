import tensorflow as tf
import cv
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#def compute_accuracy(v_xs, v_ys):
#    global prediction
#    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
#    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
#    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]    中间为X,Y方向的跨度
    # Must have strides[0] = strides[3] = 1  stride第一和最后一个值为1
    # 在卷积神经网络中，假如我们使用一个k x k的filter对m x m x d的图片进行卷积操作，strides为s，
    # 在TensorFlow中，当我们设置padding='same'时，卷积以后的每一个feature map的height和width为ceil(\frac{float(m)}{float(s)})；
    # 当设置padding='valid'时，每一个feature map的height和width为ceil(\frac{float(m-k+1)}{float(s)})。那么反过来，
    # 如果我们想要进行transposed convolution操作，比如将7 x 7 的形状变为14 x 14，那么此时，我们可以设置padding='same'，strides=2即可，与filter的size没有关系；而如果将4 x 4变为7 x 7的话，当设置padding='valid'时，即4 = ceil(\frac{7-k+1}{s})，此时s=1，k=4即可实现我们的目标。
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # 池化的核函数大小为2x2，因此ksize=[1,2,2,1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])  #最后一个通道是1代表黑白，如果是RGB则为3
# print(x_image.shape)  #[n_samples,28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32]) # patch 5x5,in_size:1,out_size:32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) #output_size：28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # output_size：28x28x32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # patch 5x5,in_size:32,out_size:64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) #output_size：14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output_size：7x7x64

# cov3 layer   it is useless
#W_conv3 = weight_variable([5,5,64,128])
#b_conv3 = bias_variable([128])
#h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
#h_pool3 = max_pool_2x2(h_conv3)

## func1 layer ##
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
# [n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


## func2 layer ##
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# the error between prediction and real data
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                              reduction_indices=[1]))       # loss
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
#if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#    init = tf.initialize_all_variables()
#else:
#    init = tf.global_variables_initializer()
# sess.run(tf.global_variables_initializer())

#for i in range(1000):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
#    if i % 50 == 0:
#        print(compute_accuracy(
#            mnist.test.images[:1000], mnist.test.labels[:1000]))

sess = tf.Session()
cross_entropy = -tf.reduce_sum(ys*tf.log(prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

with sess.as_default():

    for i in range(1000):
      batch = mnist.train.next_batch(100)
      if i%50 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            xs:batch[0], ys: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0}))