import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lstm_size = 100


class rcnn:
    def __init__(self, width, height, label_size):
        self.width = width
        self.height = height
        self.labels_size = label_size
        self.input_datas = tf.placeholder(tf.float32, shape=(None, self.width * self.height))
        self.labels = tf.placeholder(tf.float32, shape=(None, self.labels_size))
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        # self.lr = 0.01
        self.build_netz()
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.prediction)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.test_procent = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.test_acc = tf.reduce_mean(tf.cast(self.test_procent, tf.float32))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def close(self):
        self.sess.close()

    def train(self, x, y, keep_prob, lr=0.01):
        self.sess.run(self.train_step,
                      feed_dict={self.input_datas: x, self.labels: y, self.keep_prob: keep_prob, self.lr: lr})

    def test(self, x, y):
        out = self.sess.run(self.test_acc,
                            feed_dict={self.input_datas: x, self.labels: y, self.keep_prob: 1.0, self.lr: 1.0})
        return out

    def predict(self, x):
        y = np.zeros([1, 10])
        return self.sess.run(self.prediction,
                             feed_dict={self.input_datas: x, self.labels: y, self.keep_prob: 1.0, self.lr: 1.0})

    def build_netz(self):
        x = tf.reshape(self.input_datas, [-1, self.height, self.width, 1])

        cnn1 = self.layer_cnn(x, 3, 3, 1, 32, stride=1, active_fun=tf.nn.relu)
        cnn1_pool = self.layer_maxpool(cnn1, 2, stride=2)
        cnn2 = self.layer_cnn(cnn1_pool, 3, 3, 32, 64, stride=1, active_fun=tf.nn.relu)
        cnn_pool = self.layer_maxpool(cnn2, 2, stride=2)
        w = int(self.width / 4)
        h = int(self.height / 4)
        x1 = tf.reshape(cnn_pool, [-1, w * h * 64])
        #self.prediction = self.layer_fc(x1, w * h * 64, self.labels_size, tf.nn.softmax)
        x2 = self.layer_fc(x1, w * h * 64, 1000, active_fun=tf.nn.relu)
        x2_keep = tf.nn.dropout(x2, self.keep_prob)
        self.prediction = self.layer_fc(x2_keep, 1000, self.labels_size,active_fun=tf.nn.softmax)

        #self.prediction = tf.layers.dense(x2_keep, self.labels_size, activation=tf.nn.softmax)

    def build_netz2(self):
        x = tf.reshape(self.input_datas, [-1, self.height, self.width, 1])

        cnn1 = tf.layers.conv2d(x, 32, 3, 1, "SAME", activation=tf.nn.relu)
        cnn1_pool = tf.layers.max_pooling2d(cnn1, pool_size=2, strides=2, padding="SAME")
        cnn2 = tf.layers.conv2d(cnn1_pool, 64, 3, 1, "SAME", activation=tf.nn.relu)
        cnn_pool = tf.layers.max_pooling2d(cnn2, pool_size=2, strides=2, padding="SAME")
        w = int(self.width / 4)
        h = int(self.height / 4)
        x1 = tf.reshape(cnn_pool, [-1, w * h * 64])
        x2 = tf.layers.dense(x1, 1000, activation=tf.nn.relu)
        x2_keep = tf.nn.dropout(x2, self.keep_prob)
        self.prediction = tf.layers.dense(x2_keep, self.labels_size, activation=tf.nn.softmax)

    def build_netz3(self):
        x = tf.reshape(self.input_datas, [-1, self.height, self.width, 1])

        cnn1 = self.layer_cnn(x, 3, 3, 1, 32, stride=1, active_fun=tf.nn.relu)
        cnn1_pool = self.layer_maxpool(cnn1, 2, stride=2)
        cnn2 = self.layer_cnn(cnn1_pool, 3, 3, 32, 64, stride=1, active_fun=tf.nn.relu)
        cnn_pool = self.layer_maxpool(cnn2, 2, stride=2)
        w = int(self.width / 4)
        h = int(self.height / 4)
        x1 = tf.reshape(cnn_pool, [-1, w * h * 64])

        x2 = self.layer_fc(x1, w * h * 64, n_steps* lstm_size, active_fun=tf.nn.relu)
        x2_keep = tf.nn.dropout(x2, self.keep_prob)
        rx = tf.reshape(x2_keep, [-1, n_steps, lstm_size])
        rnn1 = self.layer_rnn(rx, lstm_size, tf.nn.tanh)
        self.prediction = self.layer_fc(rnn1, lstm_size, self.labels_size,tf.nn.softmax)

    def build_netz4(self):
        x = tf.reshape(self.input_datas, [-1, self.height, self.width, 1])

        cnn1 = self.layer_cnn(x, 3, 3, 1, 8, stride=1, active_fun=tf.nn.relu)
        cnn2 = self.layer_cnn(cnn1, 3, 3, 8, 32, stride=1, active_fun=tf.nn.relu)
        cnn_pool = self.layer_maxpool(cnn2, 2, stride=2)
        w = int(self.width / 2)
        h = int(self.height / 2)
        x1 = tf.reshape(cnn_pool, [-1, w * h * 32])

        self.prediction = self.layer_fc(x1, w * h * 32, self.labels_size,tf.nn.softmax)

    def layer_fc(self, input_datas, input_size, output_size, active_fun=None):
        w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[output_size]))
        out = tf.matmul(input_datas, w) + b
        if active_fun != None:
            #print("active")
            return active_fun(out)
        else:
            return out

    def layer_cnn(self, input_datas, filter_h, fiter_w, input_kernel, output_kernel, stride=1, active_fun=None):
        w = tf.Variable(tf.truncated_normal([filter_h, fiter_w, input_kernel, output_kernel], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[output_kernel]))
        out = tf.nn.conv2d(input_datas, w, [1, stride, stride, 1], "SAME") + b
        if active_fun != None:
            return active_fun(out)
        return out


    def layer_maxpool(self, input_datas, size, stride=2):
        return tf.nn.max_pool(input_datas, [1, size, size, 1], [1, stride, stride, 1], "SAME")


    def layer_rnn(self, input_datas, lstm_size, active_fun=None):
        cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        out, finalstat = tf.nn.dynamic_rnn(cell, input_datas, dtype=tf.float32)
        outputs = finalstat[1]
        if active_fun != None:
            return active_fun(outputs)
        return outputs




if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    n_batch = mnist.train.num_examples
    print("n_batch=%d" % (n_batch))
    batch_size = 100
    n_steps = 28
    n_inputs = 28
    n_outputs = 10

    netz = rcnn(n_inputs, n_steps, n_outputs)
    for i in range(20):
        for empoch in range(n_batch):
            xs, ys = mnist.train.next_batch(batch_size)
            #print(xs.shape)
            # print("empoch=%d" % (empoch))
            # out=netz.predict(xs)
            # print(out.shape)
            # break
            netz.train(xs, ys, 0.6, 0.001)
            if empoch > 0 and empoch % 100 == 0:
                xtest, ytest = mnist.test.next_batch(1000)
                acc = netz.test(xtest, ytest)
                print("i=%d; acc=%f" % (empoch, acc))
