DATA:

c10 = CIFAR-10
c100 = CIFAR-100
imgnet = ImageNet
cars = Stanford Cars (http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
birds = Birds dataset (http://www.vision.caltech.edu/visipedia/CUB-200.html)
SVHN = Google Street View House Numbers

MODELS:

Resnet110 - ResNet with 110 layers
densenet40 - DenseNet 40
resnet_wide32 - Wide ResNet 32
lenet - LeNet 5
Resnet110SD - Resnet 110 Stochastic Depth

Outputs:

logits - instead of probabilities, there are just network outputs without softmaxing. Use softmax function to get actual probabilities.

PAPERS:

ResNet - https://arxiv.org/pdf/1512.03385.pdf (Deep Residual Learning for Image Recognition)
DenseNet - https://arxiv.org/pdf/1608.06993.pdf (Densely Connected Convolutional Networks)
Wide ResNet - https://arxiv.org/pdf/1605.07146.pdf (Wide Residual Networks)
LeNet - http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf (Gradient-Based Learning Applied to Document Recognition)
ResNet SD -https://arxiv.org/pdf/1603.09382.pdf (Deep Networks with Stochastic Depth)