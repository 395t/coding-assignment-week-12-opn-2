# Open world perception - Week 12 Group 2

This week's papers are about Open world perception, particularly:
* [ Towards Open Set Deep Networks, Bendale, Boult; 2015](https://arxiv.org/abs/1511.06233)
* [	Large-Scale Long-Tailed Recognition in an Open World, Liu, Miao, Zhan, Wang, Gong, Yu; 2019](https://arxiv.org/abs/1904.05160)
* [	Class-Balanced Loss Based on Effective Number of Samples, Cui, Jia, Lin, Song, Belongie; 2019](https://arxiv.org/abs/1901.05555)
* [	Decoupling Representation and Classifier for Long-Tailed Recognition, Kang, Xie, Rohrbach, Yan, Gordo, Feng, Kalantidis; 2019](https://arxiv.org/abs/1910.09217)
* [	Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax, Li, Wang, Kang, Tang, Wang, Li, Feng; 2020](https://arxiv.org/abs/2006.10408)

# TODO: Datasets

# TODO: Towards Open Set Deep Networks

# TODO: Large-Scale Long-Tailed Recognition in an Open World

# TODO:	Class-Balanced Loss Based on Effective Number of Samples

# TODO:	Decoupling Representation and Classifier for Long-Tailed Recognition

# TODO: Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax

## Dataset

The authors of the paper are working with the long-tail [LVIS dataset](https://www.lvisdataset.org/) and their long-tail-sampled version of COCO [COCO-LT](https://arxiv.org/pdf/2007.11978.pdf). Both of these datasets are very large and take days to train on the hardware we had available (GTX 1060 6GB). The point of the paper is to work with long-tailed datasets, since the distribution of object categories in reality is typically imbalanced. The authors state in the paper that PASCAL VOC is manually balanced and thereby not suitable for this task. Since PASCAL VOC 2007 is a lot smaller than COCO, we decided to write a script to sample our own long-tailed version of VOC2007, VOC2007-LT.

TODO: image of class distribution

TODO: experiments

# TODO: Conclusion?
