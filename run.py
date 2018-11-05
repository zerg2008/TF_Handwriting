from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from PIL import Image#python3中image要从PIL中导入
#zerg，20181024 test

session = tf.InteractiveSession()

log_dir = './MNIST_LOG'    # 输出日志保存的路径
def weight_variable(shape, dtype, name):
    initial = tf.truncated_normal(shape = shape, stddev = 0.1, dtype = dtype, name = name)
    #tf.truncated_normal这个函数产生正太分布，均值和标准差自己设定。
    # shape表示生成张量的维度，mean是均值，stddev是标准差。
    return tf.Variable(initial)

def bias_variable(shape, dtype, name):
    initial = tf.constant(0.1, shape = shape, dtype = dtype, name = name)
    return tf.Variable(initial)

#定义卷积函数，其中x是输入，W是权重，也可以理解成卷积核，strides表示步长，或者说是滑动速率，包含长宽方向
#的步长。padding表示补齐数据。 目前有两种补齐方式，一种是SAME，表示补齐操作后（在原始图像周围补充0），实
#际卷积中，参与计算的原始图像数据都会参与。一种是VALID，补齐操作后，进行卷积过程中，原始图片中右边或者底部
#的像素数据可能出现丢弃的情况。
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

#这步定义函数进行池化操作，在卷积运算中，是一种数据下采样的操作，降低数据量，聚类数据的有效手段。常见的
#池化操作包含最大值池化和均值池化。这里的2*2池化，就是每4个值中取一个，池化操作的数据区域边缘不重叠。
#函数原型：def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)。对ksize和strides
#定义的理解要基于data_format进行。默认NHWC，表示4维数据，[batch,height,width,channels]. 下面函数中的ksize，
#strides中，每次处理都是一张图片，对应的处理数据是一个通道（例如，只是黑白图片）。长宽都是2，表明是2*2的
#池化区域，也反应出下采样的速度。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

mnist = input_data.read_data_sets("MNIST_DATA", one_hot = True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
# -1表示任意数量的样本数,大小为28x28的二维张量

# convolution 1
weight_conv1 = weight_variable([5, 5, 1, 32], dtype = "float", name = 'weight_conv1')
bias_conv1 = bias_variable([32], dtype = "float", name = 'bias_conv1')
hidden_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + bias_conv1)
#tf.nn.relu()函数,激活函数，是将大于0的数保持不变，小于0的数置为0
hidden_pool1 = max_pool_2x2(hidden_conv1)

# convolution 2
weight_conv2 = weight_variable([5, 5, 32, 64], dtype = "float", name = 'weight_conv2')
bias_conv2 = bias_variable([64], dtype = "float", name = 'bias_conv2')
hidden_conv2 = tf.nn.relu(conv2d(hidden_pool1, weight_conv2) + bias_conv2)
hidden_pool2 = max_pool_2x2(hidden_conv2)

# function 1
hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7 * 7 * 64])# 把池化层2的输出扁平化为1维
#形成一个 1 X（7*7*64）的矩阵
# 这里有一点想不通，通过这样的reshape不是第一个参数是匹数，就是有多少张图片，第二个参数才是
# 这张图片矩阵的行，也就是说应该是7*7*64行才对，但是怎么就变成1行，7*7*64列了？？？）
weight_fc1 = weight_variable([7 * 7 * 64, 1024], dtype = "float", name = 'weight_fc1')
bias_fc1 = bias_variable([1024], dtype = "float", name = 'bias_fc1')
hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, weight_fc1) + bias_fc1)#tf.matmul矩阵相乘
keep_prob = tf.placeholder("float")
hidden_fc1_dropout = tf.nn.dropout(hidden_fc1, keep_prob)
#对第二层卷积经过relu后的结果，基于tensor值keep_prob进行保留或者丢弃相关维度上的数据。
# 这个是为了防止过拟合，快速收敛。
#此次运算之后得到一个1 X 1024的矩阵
# function 2
weight_fc2 = weight_variable([1024, 10], dtype = "float", name = 'weight_fc2')
bias_fc2 = bias_variable([10], dtype = "float", name = 'weight_fc2')
y_fc2 = tf.nn.softmax(tf.matmul(hidden_fc1_dropout, weight_fc2) + bias_fc2)
#矩阵乘运算之后得到一个1 X 10的矩阵，再加上偏移bias_fc2，就可以进行softmax分类了，经过分类
#得到一个最大分类的概率。（在与偏移相加的时候也有一点想不通，矩阵运算结果是1行10列，
# 而偏移矩阵是10行1列，这个是怎么加的，难道tf有自适应的能力？ ）

# create tensorflow structure
cross_entropy = -tf.reduce_sum(y * tf.log(y_fc2))
tf.summary.scalar('loss', cross_entropy)

optimize = tf.train.AdamOptimizer(0.0001)
train = optimize.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_fc2, 1))
#tf.argmax(y, 1)返回向量y中最大值的索引
#tf.cast将correct_prediction的数据格式转化成dtype.
# 例如，原来correct_prediction的数据格式是bool， 这里将其转化为float
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#tf.reduce_mean为求均值，这里我理解其是对一批，比如50个样本学习后求均值，
#也就是这批样本的准确率。
tf.summary.scalar('accuracy', accuracy)
# summaries合并
merged = tf.summary.merge_all()


# initial all variables
#init = tf.initialize_all_variables()
#session = tf.Session()

# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + '/train', session.graph)
#test_writer = tf.summary.FileWriter(log_dir + '/test')
tf.global_variables_initializer().run()
#session.run(init)

# train
def Train() :
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        summary,_ = session.run([merged,train], feed_dict = {x:batch[0], y:batch[1], keep_prob:0.5})
        train_writer.add_summary(summary,i)
        if i % 100 == 0:
            print("step %4d: " % i)
            print(session.run(accuracy, feed_dict = {x:batch[0], y:batch[1], keep_prob:1}))

    print(session.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1}))
    train_writer.close()
# save variables
def save() :
	saver = tf.train.Saver()
	
	saver.save(session, save_path)

# restore variables
def restore() :
	saver = tf.train.Saver()
	saver.restore(session, save_path)

def getTestPicArray(filename):
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    im_arr = np.array(out.convert('L'))
    num0 = 0
    num255 = 0
    threshold = 100
    for x in range(x_s):#进行二值化处理
        for y in range(y_s):
            if(im_arr[x][y] > threshold):num255 = num255 + 1
            else : num0 = num0 + 1

    if(num255 > num0) :
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if(im_arr[x][y] < threshold) :  im_arr[x][y] = 0
			#if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
			#else : im_arr[x][y] = 255
			#if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

    out = Image.fromarray(np.uint8(im_arr))
    out.save("./png/2psq.png")
	#print im_arr
    nm = im_arr.reshape((1, 784))
	
	
	

    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)
	
    return nm

def testMyPicture() :
	#testNum = input("input the number of test picture:")
	for i in range(1) :
		#testPicture = raw_input("input the test picture's path:")
		oneTestx = getTestPicArray("./png/2ps.png")
		ans = tf.argmax(y_fc2, 1)
		print("The prediction answer is:") 
		print(session.run(ans, feed_dict = {x:oneTestx, keep_prob:1}))
        #意思是每个元素被保留的概率，那么 keep_prob:1就是所有元素全部保留的意思。
save_path = "network/cnn.ckpt"
Train()
#save()
#restore()
#testMyPicture()
session.close()
