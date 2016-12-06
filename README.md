# Caffe-RivaMap

This is a modified version of Caffe, a deep learning framework, that implements the following additional layers:


## Class-weighted loss layer

This layer supports class weigths in the loss function to tackle class imbalance. The code is adapted from [SegNet](https://github.com/alexgkendall/caffe-segnet/) to the most recent version of Caffe. Example use:

    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "fc7"
      bottom: "ground_truth"
      top: "loss"
      loss_param: {
        ignore_label: 0
        normalize: false
        weight_by_label_freqs: true
        class_weighting: 0
        class_weighting: 0.253
        class_weighting: 0.455
        class_weighting: 1.46
        class_weighting: 1.39
        class_weighting: 1
      }
    }


## L2-normalization layer

This layer normalizes layer activations by dividing the activations by the l2-norm. The code is adapted from the [Softmax](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/softmax_layer.cpp) and [Normalize](https://github.com/kuprel/caffe/blob/master/src/caffe/layers/normalize_layer.cpp) layers. Example use:

	layer {
	  name: "norm1"
	  type: "Normalization"
	  bottom: "conv1"
	  top: "norm1"
	}


# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
