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

Change the configs in ./tripletloss/config.py, Makesure your image path is exists, (my path is exampled)

	python train.py

I provide a pretrained example model training from a data set of 997 indentities. You could change the top fc9 layer's name and finetune this model.
But the best way is to make the model to fit your own dataset smoothly. Re-training the model on your own data set.

My approach is like the Baidu's [paper](https://arxiv.org/ftp/arxiv/papers/1506/1506.07310.pdf). (also similiar with the vgg_face's method)

Firstly, pretraining the model with softmax, or you'll get a real long period waiting for your model to converge.
Then use triplet method to finetune your model, makeing your model's output feature fits the expected Euclidean distance.

notation: maybe your need a really well cropped face dataset to do this.

If the proposal is helpful for you, please star it. thanks~
