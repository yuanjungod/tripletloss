# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.

   attention: DO NOT shuffle your input batch. 
"""
import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing

class TripletLayer(caffe.Layer):
    
    global no_residual_list,margin
    
    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        
        assert bottom[0].num % 3 == 0
        
        self.margin = layer_params['margin']
        
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        anchor_minibatch_db = []
        positive_minibatch_db = []
        negative_minibatch_db = []
        for i in range((bottom[0]).num):
            if i%3 == 0 :
                X_normalized = bottom[0].data[i].reshape(1,-1)[0]        
                anchor_minibatch_db.append(X_normalized)
            elif i%3 == 1 :
                X_normalized = bottom[0].data[i].reshape(1,-1)[0]        
                positive_minibatch_db.append(X_normalized)
            elif i%3 == 2 :
                X_normalized = bottom[0].data[i].reshape(1,-1)[0]        
                negative_minibatch_db.append(X_normalized)
        loss = 0.0
        self.no_residual_list = []
        for i in range(((bottom[0]).num)/3):
            a = np.array(anchor_minibatch_db[i]).reshape(1,-1)[0]
            p = np.array(positive_minibatch_db[i]).reshape(1,-1)[0]
            n = np.array(negative_minibatch_db[i]).reshape(1,-1)[0]
            a_p = a - p
            a_n = a - n
            ap = np.dot(a_p,a_p)
            an = np.dot(a_n,a_n)
            dist = (self.margin + ap - an)
            _loss = max(dist,0.0)
            #print ('loss:'+str(_loss)+' '+'ap:'+str(ap)+' '+'an:'+str(an))
            if _loss == 0 :
                self.no_residual_list.append(i)
                loss += _loss
        
        loss = (loss/(((bottom[0]).num)/3)/2.0)
        top[0].data[...] = loss
    

    def backward(self, top, propagate_down, bottom):

        for i in range((bottom[0]).num/3):
            if not i in self.no_residual_list:
                x_a = bottom[0].data[i*3]
                x_p = bottom[0].data[i*3+1]
                x_n = bottom[0].data[i*3+2]
                #print x_a,x_p,x_n
                bottom[0].diff[i*3] =  ((x_n - x_p)/((bottom[0]).num/3))
                bottom[0].diff[i*3+1] =  ((x_p - x_a)/((bottom[0]).num/3))
                bottom[0].diff[i*3+2] =  ((x_a - x_n)/((bottom[0]).num/3))
            else:
                bottom[0].diff[i*3] = np.zeros(shape(bottom[0].data)[1])
                bottom[0].diff[i*3+1] = np.zeros(shape(bottom[0].data)[1])
                bottom[0].diff[i*3+2] = np.zeros(shape(bottom[0].data)[1])

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass





