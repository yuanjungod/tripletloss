# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
   The layer combines the input image into triplet.Priority select the semi-hard samples
"""
import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing
import math
import config

class TripletSelectLayer(caffe.Layer):
        
    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        
        
        layer_params = yaml.load(self.param_str_)
        self.triplet = layer_params['triplet']
        
        top[0].reshape(self.triplet,shape(bottom[0].data)[1])
        top[1].reshape(self.triplet,shape(bottom[0].data)[1])
        top[2].reshape(self.triplet,shape(bottom[0].data)[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        top_archor = []
        top_positive = []
        top_negative = []
        labels = []	
        tripletlist = []
        while len(top_archor) < (self.triplet):
            a_index = random.randint(0,4)
            p_index = random.randint(0,4)
            n_index = random.randint(5,bottom[0].num-1)
            if a_index == p_index :
                p_index = max(0,a_index - 1)
                if p_index == 0 :
                    p_index = a_index + 1
                else:
                    p_index = a_index - 1
            pair = [a_index,p_index,n_index]
            if pair not in tripletlist:
                #print '===========archor_feature==========='
                archor_label = bottom[1].data[a_index]
                archor_feature = bottom[0].data[a_index].reshape(1,-1)[0]
                #print bottom[0].data[a_index]
                #print '===========positive_feature==========='
                #print positive_feature
                positive_label = bottom[1].data[p_index]
                positive_feature = bottom[0].data[p_index].reshape(1,-1)[0]
                #print '===========negative_feature==========='
                    #print negative_feature
                negative_label = bottom[1].data[n_index]
                negative_feature = bottom[0].data[n_index].reshape(1,-1)[0]

                a_p = archor_feature - positive_feature
                a_n = archor_feature - negative_feature
                #print a_p,a_n
                #print negative_label
                ap = np.dot(a_p,a_p)
                an = np.dot(a_n,a_n)
                
                if an > ap:
                    top_archor.append(archor_feature)
                    top_positive.append(positive_feature)
                    top_negative.append(negative_feature)
                    if len(top_archor) == 1:
                        print ('loss:'+'ap:'+str(ap)+' '+'an:'+str(an))
                    tripletlist.append(pair)
        
        top[0].data[...] = np.array(top_archor).astype(float)
        top[1].data[...] = np.array(top_positive).astype(float)
        top[2].data[...] = np.array(top_negative).astype(float)
    

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass







