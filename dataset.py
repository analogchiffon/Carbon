#! coding:utf-8

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')#bashに通してあるのにorz
import os
import six.moves.cPickle as pickle
import numpy as np
import cv2 as cv

class Dataset:
    def __init__(self):
        self.data_dir_path = u"/Volumes/DT01ACA200/NNCT/GS_H28/Carbon/Data/Data1_Default_int/"
        self.data = None
        self.target = None
        self.n_types_target = -1
        self.dump_name = u'dataset'
        self.scale = 128

    def get_dir_list(self, path):
        tmp = os.listdir(path)
        if tmp is None:
            return None
        return sorted([x for x in tmp if os.path.isdir(path+x)])

    def get_classid(self, fname):
        dir_list = self.get_dir_list()
        dir_name = filter(lambda x: x in fname, dir_list)
        return dir_list.index(dir_name[0])

    def load_data_target(self):

        '''
        if os.path.exists("./converted_"+str(self.scale)) is False:
            os.mkdir(u"./converted_"+str(self.scale))

        save_dir = "./converted_"+str(self.scale)
        '''

        if os.path.exists(self.dump_name):
            self.load_dataset()
            print u'Data is arleady made'
        if self.target is None:
            dir_list = self.get_dir_list(self.data_dir_path)
            ret = {}
            self.target = []
            target_name = []
            self.data = []

            for dir_name in dir_list:
                print u'Now loading ' + str(dir_name)
                sub_dir_list = self.get_dir_list(self.data_dir_path+dir_name+'/')
                for sub_dir_name in sub_dir_list:
                    file_list = os.listdir(self.data_dir_path+dir_name+'/'+sub_dir_name+'/')
                    for file_name in file_list:
                        root, ext = os.path.splitext(file_name)
                        if ext == u'.bmp':
                            abs_name = self.data_dir_path+dir_name+'/'+sub_dir_name+'/'+file_name
                            #print int(dir_name)
                            self.target.append(int(dir_name)) #教師データ(0~9(int))
                            target_name.append(str(dir_name)) #教師データディレクトリ名
                            image = cv.imread(abs_name,1)
                            hight = image.shape[0]
                            width = image.shape[1]

                            image = cv.resize(image, (self.scale,self.scale))

                            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                            #Adaptive Gaussian Thresholding
                            image = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
                            #image = image.transpose(2,0,1)#データ項目入れ替え。チャンネルを一番前に。
                            image = image/255
                            self.data.append(image) #学習データ

            self.index2name = {}
            for i in xrange(len(self.target)):
                self.index2name[self.target[i]] = target_name[i];

            print 'Data set done'

            self.data = np.array(self.data, np.float32)
            self.target = np.array(self.target, np.int32)

            self.dump_dataset();

        print u'  '+str(len(self.data))+u'pice'

    def get_n_types_target(self):
        if self.target is None:
                self.load_data_target()
        if self.n_types_target is not -1:
            return self.n_types_target

        tmp = {}
        for target in self.target:
            tmp[target] = 0
        print u'  '+str(len(tmp))+u'kinds\n'
        return len(tmp)

    def dump_dataset(self):
        pickle.dump((self.data,self.target,self.index2name), open(self.dump_name, 'wb'), -1)

    def load_dataset(self):
        self.data, self.target, self.index2name = pickle.load(open(self.dump_name, 'rb'))
