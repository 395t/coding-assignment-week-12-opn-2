from xml.etree import cElementTree as ET
import os
import random
import matplotlib.pyplot as plt
import numpy as np

import pickle
import torch

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

# borrowed from BalancedGroupSoftmax\tools\lvis_analyse.py
def get_cate_gs(class_counts):
    binlabel_count = [1, 1, 1, 1, 1]
    label2binlabel = np.zeros((5, 21), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for class_name, ins_count in class_counts:
        cid = CLASSES.index(class_name) + 1
        if ins_count < 10:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        elif ins_count < 100:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1
        elif ins_count < 1000:
            label2binlabel[3, cid] = binlabel_count[3]
            binlabel_count[3] += 1
        else:
            label2binlabel[4, cid] = binlabel_count[4]
            binlabel_count[4] += 1


    savebin = torch.from_numpy(label2binlabel).long()

    save_path = './data/VOC2007/label2binlabel.pt'
    torch.save(savebin, save_path)
    
    print("Saved label2binlabel.pt")

    # start and length
    pred_slice = np.zeros((5, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice).long()
    save_path = './data/VOC2007/pred_slice_with0.pt'
    torch.save(savebin, save_path)
    
    print("Saved pred_slice_with0.pt")

    return pred_slice

# borrowed from BalancedGroupSoftmax\tools\lvis_analyse.py
def get_split(class_counts):
    bin10 = []
    bin100 = []
    bin1000 = []
    binover = []

    for class_name, ins_count in class_counts:
        cid = CLASSES.index(class_name) + 1
        if ins_count < 10:
            bin10.append(cid)
        elif ins_count < 100:
            bin100.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['[10, 100)'] = np.array(bin100, dtype=np.int)
    splits['[100, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 21)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(21)

    split_file_name = './data/VOC2007/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)
        
    print("Saved valsplit.pkl")


directory = "./data/VOC2007/train/Annotations/"

# iterate over all annotations to count how many objects there are in the annotation
# we're doing this because we want to throw out images with few objects first as they don't affect as many other classes

filename_list = []
for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        # parse the xml file, read object names from it
        root = ET.parse(os.path.join(directory, filename))

        obj_count = 0

        for obj in root.iter("object"):
            obj_count += 1
            
        filename_list.append((filename, obj_count))
              
# sort list by object count
filename_list.sort(key = lambda x: x[1], reverse=False)
              
# dictionary containing a list for each class; each list contains the filenames of annotations with this class in it (once for every time the class appears in it)
class_dict = {}  
            
# now parse objects in files, assign filenames to class lists              
for filename, count in filename_list:
    # parse the xml file, read object names from it
    root = ET.parse(os.path.join(directory, filename))

    for obj in root.iter("object"):
        # check if the list for that class exists, make a new one if not
        obj_name = obj.find("name").text
        
        if obj_name in class_dict:
            class_dict[obj_name].append(filename)
        else:
            class_dict[obj_name] = [filename]
             
# list of class names and counts
count_list = []             
                
print("original distribution: ")
for class_name in class_dict:
    print(class_name + " count: " + str(len(class_dict[class_name])))
    
    count_list.append((class_name, len(class_dict[class_name])))
    
# sort count list
count_list.sort(key = lambda x: x[1], reverse=False)

# create desired counts
des_counts = []
for i in range(20):
    des_counts.append(int((1.415**i)/(1.415**19) * count_list[-1][1]))
    
print("Desired counts: ")
print(des_counts)

for i in range(len(des_counts)):
    # class we're dealing with now
    cur_class = count_list[i][0]

    # remove images until the actual count falls below the desired one:
    while des_counts[i] < len(class_dict[cur_class]):
        # get filename from that list
        filename = class_dict[cur_class][0]
    
        # remove filename from all class lists
        for class_name in class_dict:
            class_dict[class_name] = list(filter(lambda val: val != filename, class_dict[class_name]))
            

print("new distribution: ")
new_counts = []    
new_count_list = []
for class_name in class_dict:
    print(class_name + " count: " + str(len(class_dict[class_name])))
    new_count_list.append((class_name, len(class_dict[class_name])))
    new_counts.append(len(class_dict[class_name]))
   
print("New counts: ")
new_counts.sort()
print(new_counts)


# final list of files
file_list = []

for class_name in class_dict:
    for filename in class_dict[class_name]:
        if not filename in file_list:
            file_list.append(filename)
            
# write images to use
with open("./data/VOC2007/train/ImageSets/Main/train.txt", "w") as f:
    for filename in file_list:
        f.write(filename.replace(".xml", "") + "\n")

# write files needed for BAGS model
get_cate_gs(new_count_list)
get_split(new_count_list)
        
# plot distributions

count_list.sort(key = lambda x: x[1], reverse=True)
new_count_list.sort(key = lambda x: x[1], reverse=True)

fig, axs = plt.subplots(2)
axs[0].set_title('VOC2007 distribution')
axs[1].set_title('VOC2007-LT distribution')

word, frequency = zip(*count_list)
indices = np.arange(len(count_list))
axs[0].bar(indices, frequency, color='r')
axs[0].set_xticks(indices)
axs[0].set_xticklabels(word)
axs[0].set_xlabel('class')
axs[0].set_ylabel('examples')

word, frequency = zip(*new_count_list)
indices = np.arange(len(new_count_list))
axs[1].bar(indices, frequency, color='r')
axs[1].set_xticks(indices)
axs[1].set_xticklabels(word)
axs[1].set_xlabel('class')
axs[1].set_ylabel('examples')

plt.tight_layout()

plt.show()