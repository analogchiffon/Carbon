#! coding:utf-8

from CNN import CNN
from dataset import Dataset
from chainer import cuda
import numpy as np

cuda.init(0)

print 'load Data dataset'
dataset = Dataset()
dataset.load_data_target()
data = dataset.data
target = dataset.target
n_outputs = dataset.get_n_types_target()

model = CNN(data=data,
          target=target,
          gpu=0,
          n_outputs=n_outputs)

model.train_and_test(n_epoch=50)

model.dump_model()
