# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:32:11 2019

@author: Stpraha
"""

import os
import cv2
import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
#np.set_printoptions(threshold=np.inf) 
#from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
#from datasets.pascalvoc_common import VOC_LABELS
#    {'none': (0, 'Background'), 'aeroplane': (1, 'Vehicle'), 
#    'bicycle': (2, 'Vehicle'), 'bird': (3, 'Animal'), 
#    'boat': (4, 'Vehicle'), 'bottle': (5, 'Indoor'), 
#    'bus': (6, 'Vehicle'), 'car': (7, 'Vehicle'), 
#    'cat': (8, 'Animal'), 'chair': (9, 'Indoor'), 
#    'cow': (10, 'Animal'), 'diningtable': (11, 'Indoor'), 
#    'dog': (12, 'Animal'), 'horse': (13, 'Animal'), 
#    'motorbike': (14, 'Vehicle'), 'person': (15, 'Person'), 
#    'pottedplant': (16, 'Indoor'), 'sheep': (17, 'Animal'), 
#    'sofa': (18, 'Indoor'), 'train': (19, 'Vehicle'), 
#    'tvmonitor': (20, 'Indoor')}


annotations_path = 'F:\VOC2007\\Annotations\\'
images_path = 'F:\VOC2007\\JPEGImages\\'

voc_labels = {'none': (0, 'Background'), 'aeroplane': (1, 'Vehicle'),
              'bicycle': (2, 'Vehicle'), 'bird': (3, 'Animal'),
              'boat': (4, 'Vehicle'), 'bottle': (5, 'Indoor'),
              'bus': (6, 'Vehicle'), 'car': (7, 'Vehicle'),
              'cat': (8, 'Animal'), 'chair': (9, 'Indoor'),
              'cow': (10, 'Animal'), 'diningtable': (11, 'Indoor'),
              'dog': (12, 'Animal'), 'horse': (13, 'Animal'),
              'motorbike': (14, 'Vehicle'), 'person': (15, 'Person'),
              'pottedplant': (16, 'Indoor'), 'sheep': (17, 'Animal'),
              'sofa': (18, 'Indoor'), 'train': (19, 'Vehicle'),
              'tvmonitor': (20, 'Indoor')}

def check_data(image_set, bboxes_set, labels_set):
    if image_set[-1].shape != (300, 300, 3):
        return False
    if len(image_set) != len(bboxes_set) or len(image_set) != len(labels_set):
        return False
    if len(bboxes_set[-1][-1]) != 4:
        return False
    return True

def read_pic_batch(batch_size, batch_num):
    filenames = os.listdir(annotations_path)
    
    image_set = []
    bboxes_set = []
    labels_set = []
    for i in range(batch_num*batch_size, batch_num*batch_size+batch_size):
        j = 0
        filename = filenames[i]
        img_name = filename[:-4]

        img_file_name = images_path + img_name + '.jpg'
        img_data = cv2.imread(img_file_name)
        img_data = cv2.resize(img_data, (300, 300), interpolation=cv2.INTER_CUBIC)
        #print(img_file_name)
        image_set.append(img_data)
        
        #Read the xml
        xml_file_name = annotations_path + img_name + '.xml'
        #调用xml中的方法，将xml转换为树
        xml_tree = ET.parse(xml_file_name)
        #得到根节点
        root = xml_tree.getroot()
            
        #得到图片的尺寸
        size = root.find('size')
        #得到高、宽、channel数
        shape = [int(size.find('height').text),
                 int(size.find('width').text),
                 int(size.find('depth').text)]
        #find annotations
        bboxes = []
        labels = []
        labels_text = []
        ymin = []
        xmin = []
        ymax = []
        xmax = []
        
        for obj in root.findall('object'):
            #label:这个object的类型
            label = obj.find('name').text
            #print(label)
            #得到这个类型在VOC_LABELS里的编号
            labels.append(int(voc_labels[label][0]))
            labels_text.append(label.encode('ascii'))
                
            bbox = obj.find('bndbox')
           
            #这里也就是求出物理对于整张图片的比例          
            ymin = float(bbox.find('ymin').text) / shape[0]
            xmin = float(bbox.find('xmin').text) / shape[1]
            ymax = float(bbox.find('ymax').text) / shape[0]
            xmax = float(bbox.find('xmax').text) / shape[1]
             
            bbox = [ymin, xmin, ymax, xmax]
        
            bboxes.append(bbox)
            
        i += 1
        j += 1
        labels_set.append(labels)
        bboxes_set.append(bboxes)
               
    if not check_data(image_set, bboxes_set, labels_set):
        print('Incorrect data format')
        
    image_set = np.array(image_set, dtype = 'float32')

    return image_set, bboxes_set, labels_set

#read_pic_batch(1, 0)