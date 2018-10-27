import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from testPictiure import *

mnist = input_data.read_data_sets("MNIST_DATA", one_hot = True)
keep_prob = tf.placeholder("float")

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def testMyPicture() :
	#testNum = input("input the number of test picture:")
	for i in range(1) :
		#testPicture = raw_input("input the test picture's path:")
		oneTestx = getTestPicArray("./png/1ps.png")
		ans = tf.argmax(y_fc2, 1)
		print("The prediction answer is:")
		print(sess.run(ans, feed_dict = {x:oneTestx, keep_prob:1}))
        #意思是每个元素被保留的概率，那么 keep_prob:1就是所有元素全部保留的意思。

sess.close()