Instructions to run Balanced Group Softmax with sampled VOC:

- clone https://github.com/FishYuLi/BalancedGroupSoftmax

- copy this directory into the repo

- follow environment setup instructions on repo

- download VOC2007 and create the following file structure:

BalancedGroupSoftmax/data/VOC2007/
	test/
		Annotations
		ImageSets
		JPEGImages
		SegmentationClass
		SegmentationObject
	train/
		Annotations
		ImageSets
		JPEGImages
		SegmentationClass
		SegmentationObject
	labels.txt
	
- run VOC_sample.py

- follow training, testing instructions on repo (but test using tools/text.py)

- use https://github.com/rafaelpadilla/review_object_detection_metrics to eval using the <class id> <confidence> <left> <top> <right> <bottom> (ABSOLUTE) option