# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""Train the network."""

import caffe
from timer import Timer
import numpy as np
import os
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import sys
import config

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    """

    def __init__(self, solver_prototxt, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
	
	caffe.set_mode_gpu()
	caffe.set_device(0)
	self.solver = caffe.SGDSolver(solver_prototxt)
	if pretrained_model is not None:
	    print ('Loading pretrained model '
			   'weights from {:s}').format(pretrained_model)
	    self.solver.net.copy_from(pretrained_model)

	self.solver_param = caffe_pb2.SolverParameter()
	with open(solver_prototxt, 'rt') as f:
		pb2.text_format.Merge(f.read(), self.solver_param)


    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        """
        net = self.solver.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = (self.solver_param.snapshot_prefix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)


    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            timer.tic()
	        self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % self.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

if __name__ == '__main__':
    """Train network."""
    solver_prototxt = './solver.prototxt'
    output_dir = './models/vgg_face_tripletloss/'
    pretrained_model = './vgg_face_caffe/finetune_iters_10000.caffemodel'
    max_iters = config,MAX_ITERS
    sw = SolverWrapper(solver_prototxt, output_dir,pretrained_model)
    
    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'


















