import os
from numpy import *

# Training image data path
IMAGEPATH = '/home/seal/dataset/fast-rcnn/caffe-fast-rcnn/data/Facedevkit/tripletloss/'

# Snapshot iteration 
SNAPSHOT_ITERS = 10000

# Max training iteration
MAX_ITERS = 40000

# The number of postive samples in each minibatch, which need to be finetuned in your database
POSITIVE_NUM = 5
