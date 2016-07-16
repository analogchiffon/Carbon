#! coding:utf-8

import time
import six.moves.cPickle as pickle
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import chainer.links as L

class ImageNet(FunctionSet):
    def __init__(self, n_outputs): #入力画像/5で128,128
        super(ImageNet, self).__init__(
            conv1=L.Convolution2D(1, 32, 5, stride=1, pad=2),
            conv2=L.Convolution2D(32, 32, 5, stride=1, pad=2),
            conv3=L.Convolution2D(32, 64, 5, stride=1, pad=2),
            fc4=F.Linear(1344, 4096),
            fc5=F.Linear(4096, n_outputs),
        )

    def forward(self, x_data, y_data, train = True,gpu=-1):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)
        x = Variable(x_data)
        t = Variable(y_data)
        h = F.max_pooling_2d(F.sigmoid(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.4, train=train)
        y = self.fc5(h)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

class CNN:
    def __init__(self, data, target, n_outputs, gpu=-1):
        self.model = ImageNet(n_outputs)
        self.model_name = 'cnn_color_1_model'

        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        X = data
        y = target

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3)

        scale = 128

        #'''#
        self.X_train = self.X_train.reshape(len(self.X_train), 1, scale, scale)
        self.X_test = self.X_test.reshape(len(self.X_test), 1, scale, scale)
        #'''

        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def predict(self, x_data, gpu=-1):
        return self.model.predict(x_data, gpu)


    def train_and_test(self, n_epoch=100, batchsize=100):
        fp1 = open("accuracy.txt", "w")
        fp2 = open("loss.txt", "w")
        fp1.write("epoch\ttest_accuracy\n")
        fp2.write("epoch\ttrain_loss\n")

        epoch = 1
        best_accuracy = 0
        while epoch <= n_epoch:
            print 'epoch', epoch

            perm = np.random.permutation(self.n_train)
            sum_train_accuracy = 0
            sum_train_loss = 0
            for i in xrange(0, self.n_train, batchsize):
                X_batch = self.X_train[perm[i:i+batchsize]]
                y_batch = self.y_train[perm[i:i+batchsize]]

                real_batchsize = len(X_batch)

                self.optimizer.zero_grads()
                loss, acc = self.model.forward(X_batch, y_batch, train=True, gpu=self.gpu)
                loss.backward()
                self.optimizer.update()

                sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_train_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            print 'train mean loss={}, accuracy={}'.format(sum_train_loss/self.n_train, sum_train_accuracy/self.n_train)
            fp2.write("%d\t%f\n" % (epoch ,sum_train_loss/self.n_train))
            fp2.flush()

            # evaluation
            sum_test_accuracy = 0
            sum_test_loss = 0
            for i in xrange(0, self.n_test, batchsize):
                X_batch = self.X_test[i:i+batchsize]
                y_batch = self.y_test[i:i+batchsize]

                real_batchsize = len(X_batch)

                loss, acc = self.model.forward(X_batch, y_batch, train=False, gpu=self.gpu)

                sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            print 'test mean loss={}, accuracy={}'.format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)
            fp1.write("%d\t%f\n" % (epoch, sum_test_accuracy/self.n_test))
            fp1.flush()

            epoch += 1

        fp1.close()
        fp2.close()

    def dump_model(self):
        self.model.to_cpu()
        pickle.dump(self.model, open(self.model_name, 'wb'), -1)

    def load_model(self):
        self.model = pickle.load(open(self.model_name,'rb'))
        if self.gpu >= 0:
            self.model.to_gpu()
        self.optimizer.setup(self.model)
