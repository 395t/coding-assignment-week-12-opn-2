# Open World Perception - Group 2

This week's papers are about Open World Perception, particularly:

* [Towards Open Set Deep Networks, Bendale, Boult; 2015](https://arxiv.org/abs/1511.06233)
* [Large-Scale Long-Tailed Recognition in an Open World, Liu, Miao, Zhan, Wang, Gong, Yu; 2019](https://arxiv.org/abs/1904.05160)
* [Class-Balanced Loss Based on Effective Number of Samples, Cui, Jia, Lin, Song, Belongie; 2019](https://arxiv.org/abs/1901.05555)
* [Decoupling Representation and Classifier for Long-Tailed Recognition, Kang, Xie, Rohrbach, Yan, Gordo, Feng, Kalantidis; 2019](https://arxiv.org/abs/1910.09217)
* [Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax, Li, Wang, Kang, Tang, Wang, Li, Feng; 2020](https://arxiv.org/abs/2006.10408)

## Towards Open Set Deep Networks

For this paper there were three experiments run with various models from various repos. The datasets used are CIFAR10 and CIFAR100 in different ways to create different 'closed set' and 'open set' datasets. Experiment notebooks were implemented in colab and are included in notebooks/OSDN/. Additionally,  two failed experiments were also included, one of the official repo, and one of a Keras implementation of the official repo.

### Experiment 1
For the first experiment, CIFAR10 is used for closed data and then CIFAR100 is used as open set data. The data split between these two sets are 50/50 since the full structures are used. The repo (https://github.com/takumayagi/openmax-cifar10) uses chainer and uses a simple cnn model using only convolutional blocks with max pooling that is densely connected with drop out.

Full:
| | Accuracy | F1 |
|-|-|-|
| Openmax |  79.42 | 79.85 |
| Threshold Softmax | 62.9 | 69.7 | 

CIFAR10:
| | Accuracy | F1 |
|-|-|-|
| Openmax | 81.9 | 81.3 |
| Threshold Softmax| 92 | 85.5 |
| Softmax | 94.1 | 94.1 |


### Experiment 2
Experiment 2 is a follow up on experiment 1 developed in Keras which tests a simple canonical baseline. The way this model works is it does 11 class classification where the first 10 classes belongs to CIFAR10 and the remaining class is that of 'unknown'. The architecture of this model is a general CNN with many normalizations and horizontal flip augmentations.

| | Accuracy | F1 |
|-|-|-|
| Baseline | 81.1 | 79.8 |

### Experiment 3
For the final experiment the CIFAR100 dataset was used to create both open set and closed set data. For the closed set structure 50 random classes were used, leaving 50 for the open set. The repo (https://github.com/13952522076/Open-Set-Recognition) uses pytorch and uses the ResNet18 model. Following are the results from the experiment.

| | Accuracy | F1 |
|-|-|-|
| Openmax |  .627 | .627 |
| Threshold Softmax | 0.645 | 0.645 | 
| Softmax | 0.376 | 0.376 |

### Conclusion
In my opinion the experiment that best demonstrate the power of openmax was experiment 1. One of the biggest issues considered within openmax is how to deal with 'unknown unknowns' and one of the issues with experiment 3 is that although non of the classes are seen during training, there is still external overlap between the subcategories. For example, in CIFAR100, although there's a chance that model 3 would never have seen trout, theres a chance it would've seen flatfish, which is arguably not a complete 'unknown unknown'. Meanwhile, experiment 2 takes this completely to the other side by making all open data 'known unknowns'. This is because it's the only model that explicitly sees the open-set labels while training, and although it achieves competitive results to experiment 1, it has a much simpler task.


## Class-Balanced Loss Based on Effective Number of Samples
### **Reproducibility**
All implementation is provided in `./src/CBL/run.py`.  To reproduce the experiments, run the following in the `./src/CBL/` directory:
```shell
python run.py prep_data  # Run this once to fix TinyImagenet
python run.py train --dataset {cifar10,cifar100,tinyimagenet} --beta {None,0.9,0.99,0.999,0.9999}
python run.py plot --dataset {cifar10,cifar100,tinyimagenet}
```

### **Experiment Setup**

We fixed the long-tailed datasets and the classifier architecture across this method and the following one (Decoupling Representation and Classifier for Long-Tailed Recognition), so our results can be compared.

**Dataset** <br>
We used the following three datasets to test the method:
* **CIFAR-10**: Contains 10 classes with image size 32x32.  We preprocess this dataset to make it long-tailed. The processed training data contains 12650 examples.  Test set is unmodified.
* **CIFAR-100**: Contains 100 classes with image size 32x32.  We preprocess this dataset to make it long-tailed for a total of 16700 training examples.  Test set is unmodified.
* **TinyImagenet**: Contains 200 classes with image size 64x64.  We preprocess this dataset to make it long-tailed for a total of 33400 training examples.  Test set is unmodified.

**Model Architecture** <br>
We base our model on a ResNet50 backbone provided by PyTorch Model Zoo.  However, we modify the last layer of the ResNet model to have output dimension equal to the image dimension (e.g. `3 * 32 * 32` for CIFAR-10/100).  Then we add the final dense layer that turns this into classifier logits.   We do not use any pre-trained weights, and train all of our classifiers from scratch.

**Comparison**
We vary the most important hyperparameter $\beta$, which controls the the degree to which we account for the important of each additional example for a given class.  $\beta=0$ means we treat every example for the class as being represented by a single prototype, and the proposed method reduces to vanilla cross-entropy loss.  When $\beta\to 1$, that means we reweight each class by their counts.  Thus, $\beta$ allows us to control which point in between the above two extreme we want to be at.

Following the paper, we do a simple grid search over the values $\beta=0.9, 0.99, 0.999, 0.9999$.  We also disable class reweighting as a baseline for comparison.

### **Results: CIFAR-10**

As we can see in the following bar graph, most of the accuracy still comes from the most dominant class with largest number of examples in the dataset (orange bar).
As expected, the accuracies for underrepresented classes (green and red bars) generally improve when using class-balanced loss, although their accuracies are still significantly lower than that of the largest class.

However, we do notice that the overall accuracy over all classes (blue bar) does improve with class rebalancing, where the peak accuracy of ~40% is achieved for $\beta=0.99$.
We also observe that $\beta$ is indeed an important hyperparameter that requires tuning, since the best performance is achieved in some middle value of $\beta=0.99$.

<p float="middle">
  <img src="./src/CBL/plots/cifar10_beta_effect.png", width = "70%"/>
</p>


Below, we plot the per-class accuracy for the best performing value of $\beta$.  We can see that, even with class-rebalancing, the accuracy of **most** underrepresented classes still suffer quite a lot. We also include the class counts and per-class loss weights for reference (right).

<p float="middle">
  <img src="./src/CBL/plots/cifar10_per_class.png", width = "48%"/>
  <img src="./src/CBL/plots/cifar10_cb_weights.png", width = "48%"/>
</p>



### **Results: CIFAR-100**

We repeat the same experiment for CIFAR-10.  The general trends for overall accuracy and the largest/middle/smallest classes stay somewhat similar, except for the smallest class (red bar).

<p float="middle">
  <img src="./src/CBL/plots/cifar100_beta_effect.png", width = "70%"/>
</p>

To further investigate into what's happening across all classes, we plotted the the accuracy of every class.  Surprisingly for CIFAR-100, some of the smallest classes achieved fairly high accuracy, even though the loss weights pretty even across all classes.  We believe this is due to CIFAR-100 having 10 times more classes, and the model is simply not seeing enough of every class over the 30 epochs we train it for.  We expect this fluctuation to smooth out if we train for longer with better-tuned training hyperparameters such as learning rate, warm-up steps, and learning rate decay.  

<p float="middle">
  <img src="./src/CBL/plots/cifar100_per_class.png", width = "48%"/>
  <img src="./src/CBL/plots/cifar100_cb_weights.png", width = "48%"/>
</p>

### **Results: TinyImagenet**

Lastly, we conducted the same experiment for TinyImagenet.  This is arguably the most challenging dataset we tried, with 200 classes and larger images. As shown below, the class imbalance issue was much more pronounced for ImageNet, presumably because of even larger number of classes.

<p float="middle">
  <img src="./src/CBL/plots/tinyimagenet_beta_effect.png", width = "70%"/>
</p>

Overall, the plots look similar to those for CIFAR-100.

<p float="middle">
  <img src="./src/CBL/plots/tinyimagenet_per_class.png", width = "48%"/>
  <img src="./src/CBL/plots/tinyimagenet_cb_weights.png", width = "48%"/>
</p>



### **Conclusions**
From the above experiments, we drew the following conclusions:
* For class-balanced loss, $\beta$ really needs to be tuned for each dataset.
* Loss weights see a significant change only for extremely small classes.  Once a class has over ~100 examples, its weight is essentially equal to that of the largest class.
* It is important to train the model for sufficient number of epochs to ensure that the model is given enough exposure to the smallest classes.


## Decoupling Representation and Classifier for Long-Tailed Recognition

### **Code Structure**

The code needed to replicate these experiments are found in ``./notebooks/DecouplingRepresentationAndClassifer/*``. There is a seperate notebook for each of the datasets that the experiments were tested on. The code for loading the datasets, defining the model architectures, and the training code is found in these notebooks. 

### **Command to Run the Code**

The code for the experiemnts are in a ``.ipynb notebook`` that was created and run on Colab. It includes the necessary commands to run directly on Colab. If trying to replicate these experiments, I recommend directly uploading the notebooks to Colab and running it there to avoid dealing with packages and dependencies. 

### **Datasets**

We ran experiments on the three datasets:

* **CIFAR-10**  32x32, 10 Classes, 5000 Training Images, 1000 Test Images per Class  https://www.cs.toronto.edu/~kriz/cifar.html
* **CIFAR-100**  32x32, 100 Classes, 500 Training Images, 100 Test Images per Class https://www.cs.toronto.edu/~kriz/cifar.html
* **TinyImageNet**  64x64, 200 Classes, 500 Training, 50 Test Images per Class http://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf

Normally, the training data for these datasets is balanced and has equal number of images per class. However, the paper for these experiments deals with long-tailed data distributions. Therefore, we took these three datasets and created long-tailed versions of these datasets that we will refer to as Long-Tailed CIFAR-10, Long-Tailed CIFAR-100, and Long-Tailed Tiny ImageNet. We created these new training datasets by sampling fewer and fewer images for successive class labels. For instance, the data frequency distribution for our Long-Tailed CIFAR-10 dataset looks like this:

<p float="middle">
  <img src="./notebooks/DecouplingRepresentationAndClassifer/Images/CIFARDataDistribution.PNG", width = "50%"/>
</p>

### **Task**

Evaluating Joint Training vs Decoupling Learning Representation and Classifier Methods for Long-tailed Recognition

### **Model Architecture**

We use the pytorch implementation of the [ResNeXt-50 architecture](https://pytorch.org/hub/pytorch_vision_resnext/) for the backbone network that learns the feature representations of the images. For the Joint approach, another linear layer is appended to the end of the backbone network such that for an image X, I obtain a linear classifier g(z) = W^T + b, where W denotes the linear layer weights, b is the bias, and z is the learned feature representation of the image X found by the ResNeXt-50 model. In this method, the gradient is passed through the classifier and the representation learning network and they are jointly trained. 

For the decoupled approach, we trained the backbone ResNeXt-50 model for some number of epochs and then discard its classifier and only keep the components of the network that are important for feature representation. Then, we load the pre-trained backbone network and initialize a new linear layer that is fine tuned for a fewer number of epochs while keeping the backbone network fixed. The linear classifier layer is trained with gradient updates and the goal is to better define decision boundaries.

### **Conducted Experiments**

The paper explains two different approaches:
* Joint - The Backbone Feature Representation Network and the Classifer are Trained Together
* Decoupled - The Backbone Feature Representation Network is fixed while the Classifier is Fine-tuned and re-balanced

In addition, we test out the performance of the two approaches when using different sampling strategies:
* Instance-balanced - each image has an equal chance of being selected
* Class-balanced - each class is equally likely to be selected, and each image in the class is equally likely to be selected
* Square Root - Variant of instance-balanced
* Progressively-balanced - Combination of both Instance-balanced and Class-balanced

With the decoupled approach, we tried 2 different types of classifiers:
* Classifier Re-Training (cRT) - Standard approach of reinitializing classifier and fine-tuning it with fixed backbone network
* Tau-normalized Classifier (Tau-Norm) - approach where weights of linear layer are adjusted through a tau-normalization procedure where tau is a hyperaparameter

We conduct experiments on the different sampling strategies and how they affect performance for both the Joint and Decoupled Approaches.

### **Results**

**Long-tailed CIFAR-10**

|             | Joint         | cRT          | Tau-Norm      | 
| ----------- | ------------- | ------------ | ------------- |
| Instance-balanced   | 0.4049           | 0.4531          | 0.4501            | 
| Class-balanced      | 0.405        | 0.4489           | 0.4421            |
| Square Root         | 0.4343  | 0.4698  | 0.4522   |
| Progressively-balanced   | 0.4209 | 0.4400 | 0.4479   |


<p float="middle">
  <img src="./notebooks/DecouplingRepresentationAndClassifer/Images/CIFAR10Results.PNG", width = "50%"/>
</p>

The Joint approach was trained for 30 epochs, learning rate = 1e-3, and a batch size = 128, using Cross Entropy Loss and Adam Optimizer. For the Decoupled methods, the backbone network was trained for 30 epochs, and the classifiers were re-trained for 10 epochs with the same batch size, learning rate, and other training/optimization parameters as the Joint approach. The Tau hyperparameter was set to 0.2.

The Decoupled method outperforms the Joint training approach on the Long-tailed Variant of CIFAR-10 by about 5%, especially with the cRT method. With the Joint training approach, the sampling methods were important to obtain good results as Square-root sampling led to an approximate 3% performance gain over instance-balanced sampling. In our experiments with CIFAR-10, the cRT classifier outperformed the Tau-Norm classifier. Unlike the results in the paper, for this dataset, we found that Square-root sampling outperformed instance-balanced sampling for both cRT and Tau-Norm classifiers. Overall, decoupling the representation network and the classifier led to a significant performance gain for this particular dataset.

**Long-tailedCIFAR-100**

|             | Joint         | cRT          | Tau-Norm      | 
| ----------- | ------------- | ------------ | ------------- |
| Instance-balanced   | 0.2690          | 0.3245          | 0.3224            | 
| Class-balanced      | 0.2708        | 0.3017          | 0.3062            |
| Square Root         | 0.3089 | 0.3204  | 0.3165   |
| Progressively-balanced   | 0.2955 | 0.3150 | 0.3107   |


<p float="middle">
  <img src="./notebooks/DecouplingRepresentationAndClassifer/Images/CIFAR100Results.PNG", width = "50%"/>
</p>

The Joint approach was trained for 30 epochs, learning rate = 1e-3, and a batch size = 128, using Cross Entropy Loss and Adam Optimizer. For the Decoupled methods, the backbone network was trained for 30 epochs, and the classifiers were re-trained for 10 epochs with the same batch size, learning rate, and other training/optimization parameters as the Joint approach. The Tau hyperparameter was set to 0.5.

Like the previous dataset, we observe a significant performance gain when using the paper's suggested decoupling approach. Specifically, we see a about a 6% top-1 accuracy gain over the Joint approach. With the Joint-approach, square root sampling was the best option. However, with the decoupled classifiers cRT and Tau-Norm, we found that the instance-balanced sampling methods were more successful compared to the other 3 sampling strategies. This aligns with the paper's argument that with the decoupling approach, the model learns to generalize better, which makes the usage non-instance-balanced sampling obsolete. All in all, we were able to get results for this dataset that concur with the points made in the paper. 

**Long-Tailed Tiny-ImageNet**

|             | Joint         | cRT          | Tau-Norm      | 
| ----------- | ------------- | ------------ | ------------- |
| Instance-balanced   | 0.1875          | 0.2918          | 0.2816            | 
| Class-balanced      | 0.1799        | 0.277          | 0.2852            |
| Square Root         | 0.1904 | 0.2861  | 0.2746   |
| Progressively-balanced   | 0.1844 | 0.2746 | 0.2722   |


<p float="middle">
  <img src="./notebooks/DecouplingRepresentationAndClassifer/Images/TinyImageNetResults.PNG", width = "50%"/>
</p>

The Joint approach was trained for 5 epochs, learning rate = 1e-3, and a batch size = 128, using Cross Entropy Loss and Adam Optimizer. For the Decoupled methods, the backbone network was trained for 5 epochs, and the classifiers were re-trained for 2 epochs with the same batch size, learning rate, and other training/optimization parameters as the Joint approach. The Tau hyperparameter was set to 0.7.

Similar to the other two datasets, the decoupled methods outperform the joint training approach on the long-tailed data. cRT instance-balanced classifier was able to get a top-1 accuracy that is approximately 10% more than the best Jointly trained classifier. In addition, the instance-balanced methods performed relatively well and were able to sufficiently generalize. The cRT and Tau-Norm classifiers demonstrated similar top-1 accuracies. Overall, we were able to get a significant performance gain with the the decoupled approach on this dataset when compared to the joint approach.

**Sampling Strategies Ablation**

We typically found that the Square-root sampling approach was most beneficial for the Joint method, while, the samping method did not play that large of a factor with cRT and Tau-norm classifiers. When training jointly, other sampling methods outperformed instance-balanced sampling. However, decoupling the representation learning and classifier led to instance-balanced sampling typically leading to the best results.

### **Conclusion**

All in all, the decoupled methods led to significant performance gains over the jointly trained approach. This is evident for all three datasets that we performed experiments on. We found tha the sampling method used is important when training jointly, while it is not so important when using the decoupled methods as the instance-balanced sampling led to competitive if not better results than the latter approaches. We found that the decoupled approach led to better performance on long-tailed image recognition task. The decoupled approach allowed us to re-balance the classifiers while keeping the backbone represenation network fixed. By using relatively simple methods, we were able to see that the decoupled approach is a legitimate option in improving performance of models on long-tailed datasets.

<!-- # TODO: Towards Open Set Deep Networks -->
# Large-Scale Long-Tailed Recognition in an Open World

## Experimental Setup
Code for the OLTR experiments are directly taken from the authors original implementation ([Github Source](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)). All experiments are done on single V100 GPU. We experiment OLTR on following 2 datasets:
* **Places_LT**: Long Tailed version of [Places365](http://places2.csail.mit.edu/download.html) dataset prepared by the authors for experiments in the paper
* **Tiny-ImageNet_LT**: Long Tailed version of [Tiny-ImageNet-200](https://www.kaggle.com/c/tiny-imagenet) dataset prepared by us by sampling a subset of the dataset following a Pareto distribution (alpha=6)

Places_LT | Tiny-ImageNet_LT
:---:|:---:
<img src='src/OLTR/plots/places_trn_dist.png' height="300">|<img src='src/OLTR/plots/tin_trn_dist.png' height="300">

## Results
### Places_LT
We can see that many-shot labels heavily contribute to the overall validation accuracy in stage 1 training. In stage 2, we can observe that while many-shot labels are still the highest contributor to the overall validation accuracy, median and low-shot labels start performing comparable.
<p align="center"><img src='src/OLTR/plots/places_trn_loss_acc_stage1.png' height="250"> <img src='src/OLTR/plots/places_val_acc_stage1.png' height="250"></p>
<p align="center">Fig.1 Stage 1 training statistics</p>
<p align="center"><img src='src/OLTR/plots/places_trn_loss_acc_stage2.png' height="250"> <img src='src/OLTR/plots/places_val_acc_stage2.png' height="250"></p>
<p align="center">Fig.2 Stage 2 training statistics</p>

### Tiny-ImageNet_LT
We see a similar trend in the overall validation accuracy's distribution over many, median and low shot label regimes across stage 1 and stage 2
<p align="center"><img src='src/OLTR/plots/tin_trn_loss_acc_stage1.png' height="250"> <img src='src/OLTR/plots/tin_val_acc_stage1.png' height="250"></p>
<p align="center">Fig.3 Stage 1 training statistics</p>
<p align="center"><img src='src/OLTR/plots/tin_trn_loss_acc_stage2.png' height="250"> <img src='src/OLTR/plots/tin_val_acc_stage2.png' height="250"></p>
<p align="center">Fig.4 Stage 2 training statistics</p>

### Test Results (*Top 1 Accuracy*)

Dataset | Overall | Many-shot | Median-shot | Low-shot | Open
:---:|:---:|:---:|:---:|:---:|:---:
Tiny-ImageNet_LT | 0.228 | 0.285 | 0.247 | 0.157 | -
Places_LT | 0.389 | 0.435 | 0.403 | 0.279 | -
Places_LT open | 0.385 | 0.433 | 0.399 | 0.274 | 0.082

<!-- # TODO:	Class-Balanced Loss Based on Effective Number of Samples -->

<!-- # TODO:	Decoupling Representation and Classifier for Long-Tailed Recognition -->

# Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax

## Dataset

The authors of the paper are working with the long-tail [LVIS dataset](https://www.lvisdataset.org/) and their long-tail-sampled version of COCO [COCO-LT](https://arxiv.org/pdf/2007.11978.pdf). Both of these datasets are very large and take days to train on the hardware we had available (GTX 1060 6GB). The point of the paper is to work with long-tailed datasets, since the distribution of object categories in reality is typically imbalanced. The authors state in the paper that PASCAL VOC is manually balanced and thereby not suitable for this task. Since PASCAL VOC 2007 is a lot smaller than COCO, we decided to write a script (``VOC_sample.py``) to sample our own long-tailed version of VOC2007.

<p float="middle">
  <img src="./src/BalancedGroupSoftmax/VOC-LT_dist.png", width = "100%"/>
</p>

The number of instances per class are now as follows:

|Class name|Instances|
|---|---|
|person|3971|
|car|1340|
|chair|683|
|bird|588|
|pottedplant|521|
|dog|507|
|bottle|451|
|bicycle|338|
|sofa|331|
|horse|239|
|boat|168|
|motorbike|118|
|cat|83|
|tvmonitor|59|
|cow|31|
|sheep|19|
|aeroplane|15|
|train|13|
|bus|7|
|diningtable|5|

The full VOC2007 test set was used for testing.

## Experiment

[The official repo](https://github.com/FishYuLi/BalancedGroupSoftmax) was used for the experiments. It provides config files and models pretrained on COCO for many different architectures. Faster R-CNN using a ResNet-50 backbone was chosen for testing bounding box based object detection. The pretrained model is trained on the dataset for 12 epochs using SGD and a learning rate of 0.00125 to create a baseline model. Then, it is augmented with the group softmax and the classification layer is trained for another 12 epochs. The files ``label2binlabel.pt, pred_slice_with0.pt and valsplit.pkl`` have to be generated based on the class distribution. This is done by the same script that samples the dataset. The training takes about 6 hours for the baseline model and 2.5 for the bags model on a GTX 1060 GPU. The config files for the baseline and the bags model are provided in ``src/BalancedGroupSoftmax/configs``.

The testing script had to be modified slightly to use the VOC style dataset instead of LVIS. It outputs the predictions in text files in the ``results`` directory. The modified script is provided in ``src/BalancedGroupSoftmax/test.py``. The predictions can then be evaluated using [this object detection metrics tool](https://github.com/rafaelpadilla/review_object_detection_metrics).

## Results

The results are underwhelming to say the least. The baseline model outperforms the group softmax model even on the tail classes:

|Model|mAP|
|---|---|
|baseline|0.755|
|BAGS|0.719|
 
 |Class|baseline AP|BAGS AP|
 |---|---|---|
 |person|0.872|0.855|
|car|0.824|0.785|
|chair|0.506|0.466|
|bird|0.762|0.730|
|pottedplant|0.589|0.551|
|dog|0.875|0.860|
|bottle|0.684|0.644|
|bicycle|0.844|0.822|
|sofa|0.721|0.718|
|horse|0.884|0.865|
|boat|0.614|0.558|
|motorbike|0.850|0.821|
|cat|0.896|0.888|
|tvmonitor|0.798|0.759|
|cow|0.750|0.730|
|sheep|0.718|0.700|
|aeroplane|0.837|0.819|
|train|0.812|0.756|
|bus|0.785|0.758|
|diningtable|0.480|0.290|

We assume that a mistake was made while adapting the architecture to the new dataset but ran out of time trying to find it.

<!-- # TODO: Conclusion? -->
