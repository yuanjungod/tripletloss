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
		for i in range(5):
			# archor select
			archor_label = bottom[1].data[i]
			archor_feature = bottom[0].data[i].reshape(1,-1)[0]
			#print '===========archor_feature==========='
			#print archor_feature
			for j in range(5):
				# positive select
				positive_label = bottom[1].data[j]
				positive_feature = bottom[0].data[j].reshape(1,-1)[0]
				#print '===========positive_feature==========='
				#print positive_feature
				if positive_label == archor_label and not j == i:
					for n in range(5,(bottom[0]).num):
						# negative select
						negative_label = bottom[1].data[n]
						negative_feature = bottom[0].data[n].reshape(1,-1)[0]
						#print '===========negative_feature==========='
						#print negative_feature
						if not negative_label == archor_label:
							a_p = archor_feature - positive_feature
							a_n = archor_feature - negative_feature
							#print a_p,a_n
							#print negative_label
							ap = np.dot(a_p,a_p)
							an = np.dot(a_n,a_n)			    
							if an > ap :
								top_archor.append(archor_feature)
								top_positive.append(positive_feature)
								top_negative.append(negative_feature)
								#print ('loss:'+'ap:'+str(ap)+' '+'an:'+str(an))
								if len(top_archor) == self.triplet:
									break
				if len(top_archor) == self.triplet:
					break
			if len(top_archor) == self.triplet:
				break
		if len(top_archor)<self.triplet:
			for i in range(5):
				# archor select
				archor_label = bottom[1].data[i]
				archor_feature = bottom[0].data[i].reshape(1,-1)[0]
				#print '===========archor_feature==========='
				#print archor_feature
				for j in range(5):
					# positive select
					positive_label = bottom[1].data[j]
					positive_feature = bottom[0].data[j].reshape(1,-1)[0]
					#print '===========positive_feature==========='
					#print positive_feature
					if positive_label == archor_label and not j == i:
						for n in range(5,(bottom[0]).num):
							# negative select
							negative_label = bottom[1].data[n]
							negative_feature = bottom[0].data[n].reshape(1,-1)[0]
							#print '===========negative_feature==========='
							#print negative_feature
							if not negative_label == archor_label:
								top_archor.append(archor_feature)
								top_positive.append(positive_feature)
								top_negative.append(negative_feature)

								if len(top_archor) == self.triplet:
									break
					if len(top_archor) == self.triplet:
						break
				if len(top_archor) == self.triplet:
					break
		#print shape(top_archor),shape(top_positive),shape(top_negative),shape(top_weights)
		top[0].data[...] = np.array(top_archor).astype(float)
		top[1].data[...] = np.array(top_positive).astype(float)
		top[2].data[...] = np.array(top_negative).astype(float)
    

    def backward(self, top, propagate_down, bottom):
	"""This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass







