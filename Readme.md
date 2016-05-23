# TripletLoss Example

This aims to reproduce the loss function used in Google's [FaceNet paper](http://arxiv.org/abs/1503.03832v1).

This is just an example which shows how to use the method to train a model.

## Online sample selection

I provide an online triplet sample selection usage. Just a runnable strategy. YOU MAY CHANGE IT to fit your own strategy.

## Setup

Rebuild your caffe directory and makesure your python could find the added layers.

Go to your caffe root path:
	
	cp Makefile.configexample Makefile.config
	
Open Makefile.config uncommit the line :

	WITH_PYTHON_LAYER := 1
	
Then return to caffe root create build directory:

	mkdir build
	cd build
	cmake ..
	make all & make pycaffe
    
## Usage

(You should review the code first~)

a.Pre-training your model with softmax loss. Initialize the parameters of the CNN(layer's lr_mult,decay_mult) which you find in your caffe folder. Don't forget the 'base_lr' in solver.prototxt.

b.Uncommit the lines of 'softmax' in train.prototxt and commit 'tripletloss'.

c.Change the configs in ./tripletloss/config.py, Makesure your image path is exists, (my path is exampled)

	python train.py

d.Uncommit the lines of 'tripletloss' in train.prototxt and commit 'softmax'.

e.Initialize the parameters mentioned above.

	python train.py
	
I provide a pretrained example model training from a data set of 997 indentities. You could change the top fc9 layer's name and finetune this model.
But the best way is to make the model to fit your own dataset smoothly. Re-training the model on your own data set.

My approach is like the Baidu's [paper](https://arxiv.org/ftp/arxiv/papers/1506/1506.07310.pdf). (also similiar with the vgg_face's method)

Firstly, pretraining the model with softmax, Here means to fix your featrue out put to satisfy the classification model.
Then using triplet method to train the linear transform layer fc9_1, making the feature's affine projection fits the expected Euclidean distance.

notation: maybe your need a really well cropped face dataset to do this.

If the proposal is helpful for you, please star it. thanks~
