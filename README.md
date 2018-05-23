# Calibration of Neural Networks
## Introduction
This repository contains all scripts needed to train neural networks (ResNet, DenseNet, DAN etc) and to calibrate the probabilities. These networks are trained on 4 different datasets and the model weights and output logits are available for use in this repository.

## Structure
Structure of the repository:
- [Logits](logits) - pickled files with logits for the trained models.
- [Models](models) - model weights of the trained models.
- [Reliability diagrams](reliability_diagrams) - reliability diagrams generated for the models.
- [Scripts](scripts) - Python code and notebooks used to train the models, evaluate the outcome and calibrate the probabilities of the models (Python 3.6.4, Keras 2.1.4, Tensorflow 1.4.1)

## Datasets

Following datasets were used:
- CIFAR-10/100 - more information on https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet - more information on http://www.image-net.org/challenges/LSVRC/2012/
- SVHN - more information on http://ufldl.stanford.edu/housenumbers/
- Caltech-UCSD Birds - more information on http://www.vision.caltech.edu/visipedia/CUB-200.html

## Models
Following models were used and trained:
- ResNet - based on paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385)
- ResNet (SD) - based on paper ["Deep Networks with Stochastic Depth"](https://arxiv.org/abs/1603.09382)
- Wide ResNet - based on paper ["Wide Residual Networks"](https://arxiv.org/abs/1605.07146)
- DenseNet - based on paper ["Densely Connected Convolutional Networks"](https://arxiv.org/abs/1608.06993)
- LeNet - based on paper ["Gradient-based learning applied to document recognition"](https://ieeexplore.ieee.org/document/726791/)
- DAN - based on paper ["Deep Unordered Composition Rivals Syntactic Methods for Text Classification"](https://www.researchgate.net/publication/301404438_Deep_Unordered_Composition_Rivals_Syntactic_Methods_for_Text_Classification)

The hyperparameters and data preparation suggested by the authors of the papers were used to train the models, except for LeNet and DAN.

## Calibration
Following calibration methods were used:
- Histogram binning - based on paper ["Obtaining calibrated probability estimates from
decision trees and naive bayesian classifiers"](https://dl.acm.org/citation.cfm?id=655658)
- Isotonic regression - based on paper ["Transforming classifier scores into accurate multiclass probability estimates
"](https://dl.acm.org/citation.cfm?id=775151)
- Temperature Scaling - based on paper ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599)
- Beta Calibration - based on paper ["Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"](http://proceedings.mlr.press/v54/kull17a.html)

## Author
	Markus Kängsepp, University of Tartu
