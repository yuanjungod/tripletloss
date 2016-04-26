import os
from numpy import *

# Training image data path
IMAGEPATH = ''

# Training sample image list data path
SAMPLEPATH = ''

# Snapshot iteration 
SNAPSHOT_ITERS = 10000

# Max training iteration
MAX_ITERS = 400000

# The number of postive samples in each minibatch, which need to be finetuned in your database
POSITIVE_NUM = 5