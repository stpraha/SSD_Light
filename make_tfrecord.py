# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:58:49 2018

@author: Stpraha
"""
import os
import cv2
import tensorflow as tf
#import numpy as np
import xml.etree.ElementTree as ET

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

#dataset_dir = 'F:\\VOC2007\\'
output_dir = 'C:\\Users\\Stpraha\\SSD_Light\\tfrecords\\'

dataset_annotations = 'F:\\VOC2007\\Annotations\\'
#dataset_imagesets = 'ImageSets\\'
dataset_images = 'F:\\VOC2007\\JPEGImages\\'

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

#print(VOC_LABELS)

def make_tf_records():
    path = dataset_annotations
    filenames = os.listdir(path)
    #print(filenames)
    
    i = 0
    fidx = 0
    while i < len(filenames):
    #while i < 100:
        print('now converting pic No.', i)
             
        if i % 64 == 0:
            tf_filename = output_dir + 'voc_train_%03d.tfrecord' % (fidx)  

            #创建.tfrecord文件，准备写入
            tfrecord_write = tf.python_io.TFRecordWriter(tf_filename)
            fidx += 1
            
        j = 0
        filename = filenames[i]
        img_name = filename[:-4]
     
        #Read the image
        #这里不用tf.gfile.FastGFile()，因为在差值的时候可能会出现bug
        #改用openCV
        img_file_name = dataset_images + img_name + '.jpg'
        #img_data = tf.gfile.FastGFile(img_file_name, 'rb').read()
        img_data = cv2.imread(img_file_name)
        img_data = cv2.resize(img_data, (300, 300), interpolation=cv2.INTER_CUBIC)
        img_data = img_data.tobytes()
        #plt.imshow(img_data)
        print(len(img_data))
        
        
        #Read the xml
        xml_file_name = dataset_annotations + img_name + '.xml'
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
        #print(shape[0], shape[1], shape[2])
        #find annotations
        bboxes = []
        labels = []
        labels_text = []
        difficult = []
        truncated = []
        ymin = []
        xmin = []
        ymax = []
        xmax = []
            
        for obj in root.findall('object'):
            #label:这个object的类型
            label = obj.find('name').text
            #得到这个类型在VOC_LABELS里的编号
            labels.append(int(voc_labels[label][0]))
            labels_text.append(label.encode('ascii'))
            #print(int(VOC_LABELS[label][0]), VOC_LABELS[label][1])  
            #print(obj.find('difficult').text)
            if obj.find('difficult').text == '1':
                #标为difficult的目标在测试成绩的评估中一般会被忽略
                #print(obj.find('difficult').text)
                difficult.append(int(obj.find('difficult').text))
            else:
                difficult.append(0)
                #print(obj.find('difficult').text)
                
            if obj.find('truncated').text == '1':
                #标为truncated的目标说明没有被框完整
                truncated.append(int(obj.find('truncated').text))
            else:
                truncated.append(0)          
            #print(img_name, truncated) 
                
            bbox = obj.find('bndbox')
                
            #这里也就是求出物理对于整张图片的比例          
            ymin.append(float(bbox.find('ymin').text) / shape[0])
            xmin.append(float(bbox.find('xmin').text) / shape[1])
            ymax.append(float(bbox.find('ymax').text) / shape[0])
            xmax.append(float(bbox.find('xmax').text) / shape[1])
                
        bboxes = (ymin, xmin, ymax, xmax)
        #print(labels)  
        #image_data, shape, bboxes, labels, labels_text, difficult, truncated
                
        #for b in bboxes: 
        #如通b的长度不为4，则报错了
        #assert len(b) == 4
        #[l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
      
        img_format = b'JPEG'
        
        #https://www.jianshu.com/p/78467f297ab5
        example = tf.train.Example(features = tf.train.Features(feature = {
                'image/height': tf.train.Feature(int64_list = tf.train.Int64List(value = [shape[0]])),
                'image/width': tf.train.Feature(int64_list = tf.train.Int64List(value = [shape[1]])),
                'image/channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [shape[2]])),
                'image/shape': tf.train.Feature(int64_list = tf.train.Int64List(value = shape)),
                'image/object/bbox/ymin': tf.train.Feature(float_list = tf.train.FloatList(value = ymin)),
                'image/object/bbox/xmin': tf.train.Feature(float_list = tf.train.FloatList(value = xmin)),
                'image/object/bbox/ymax': tf.train.Feature(float_list = tf.train.FloatList(value = ymax)),
                'image/object/bbox/xmax': tf.train.Feature(float_list = tf.train.FloatList(value = xmax)),
                'image/object/bbox/label': tf.train.Feature(int64_list = tf.train.Int64List(value = labels)),
                'image/object/bbox/label_text': tf.train.Feature(bytes_list = tf.train.BytesList(value = labels_text)),
                'image/object/bbox/difficult': tf.train.Feature(int64_list = tf.train.Int64List(value = difficult)),
                'image/object/bbox/truncated': tf.train.Feature(int64_list = tf.train.Int64List(value = truncated)),
                'image/format': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_format])),
                'image/encoded': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_data]))}))

        tfrecord_write.write(example.SerializeToString())
        
        i += 1
        j += 1
  
    print('Finish converting DATASET to TFRECORDS')

#make_tf_records()