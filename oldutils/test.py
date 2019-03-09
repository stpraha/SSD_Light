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

images_path = 'F:\VOC2007\\JPEGImages\\'
annotations_path = 'F:\VOC2007\\Annotations\\'


def read_pic_batch(batch_size, batch_num):

    filenames = os.listdir(annotations_path)
    
    image_set = []

    for i in range(batch_num*batch_size, batch_num*batch_size+batch_size):
        filename = filenames[i]
        img_name = filename[:-4]

        img_file_name = images_path + img_name + '.jpg'
        img_data = cv2.imread(img_file_name)

        img_data = cv2.resize(img_data, (300, 300), interpolation=cv2.INTER_CUBIC)

        image_set.append(img_data)
        print(img_file_name)

    return image_set

read_pic_batch(1, 0)


