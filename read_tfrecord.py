# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:36:15 2018

@author: Stpraha
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def read_from_tfrecords_slim(path, tfrecord_filename):
    provider = slim.dataset_data_provider.DatasetDataProvider()
    



def read_from_tfrecords_tf(path, tfrecord_filename):
    """
    Argument:
        path: The tfrecord floder
        tfrecord_filename: the name
    Return:
        img_feature: a list of image and its feature read from tfrecord file.
    """
    image_set = []
    bboxes_set = []
    labels_set = []

    tfrecord_file_path = path + tfrecord_filename
    reader = tf.TFRecordReader()
    file_queue = tf.train.string_input_producer([tfrecord_file_path], num_epochs = None)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example,
                                           features = {
                                                   'image/height': tf.FixedLenFeature([1], tf.int64),
                                                   'image/width': tf.FixedLenFeature([1], tf.int64),
                                                   'image/channels': tf.FixedLenFeature([1], tf.int64),
                                                   'image/shape': tf.FixedLenFeature([3], tf.int64),
                                                   'image/object/bbox/ymin': tf.VarLenFeature(dtype = tf.float32),
                                                   'image/object/bbox/xmin': tf.VarLenFeature(dtype = tf.float32),
                                                   'image/object/bbox/ymax': tf.VarLenFeature(dtype = tf.float32),
                                                   'image/object/bbox/xmax': tf.VarLenFeature(dtype = tf.float32),
                                                   'image/object/bbox/label': tf.VarLenFeature(dtype = tf.int64),
                                                   'image/object/bbox/label_text': tf.VarLenFeature(dtype = tf.string),
                                                   'image/object/bbox/difficult': tf.VarLenFeature(dtype = tf.int64),
                                                   'image/object/bbox/truncated': tf.VarLenFeature(dtype = tf.int64),
                                                   'image/format': tf.FixedLenFeature((), tf.string, default_value = 'JPEG'),
                                                   'image/encoded': tf.FixedLenFeature((), tf.string, default_value = '')})
    
    img_data = tf.decode_raw(features['image/encoded'], tf.uint8)
    img_data = tf.reshape(img_data, [300, 300, 3])

    ymin_data = tf.cast(features['image/object/bbox/ymin'], tf.float32)
    xmin_data = tf.cast(features['image/object/bbox/xmin'], tf.float32)
    ymax_data = tf.cast(features['image/object/bbox/ymax'], tf.float32)
    xmax_data = tf.cast(features['image/object/bbox/xmax'], tf.float32)
    
    label_data = tf.cast(features['image/object/bbox/label'], tf.int64)

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
            
        for i in range(32):                                     ######################## 这  里  记  得  改 ###############################
            image, ymin, xmin, ymax, xmax, label = sess.run([img_data, ymin_data, xmin_data, ymax_data, xmax_data, label_data])#在会话中取出image和bbox
            #print('image', image)
            bboxes_set.append([ymin, xmin, ymax, xmax])
            labels_set.append(label)
            image_set.append(image)
                
        coord.request_stop()
        coord.join(threads)
        #将image进行标准化
        
    images = np.array(image_set, dtype = 'float32')
        
        #bboxes_nums = []
#        bboxes = []
#        for i, bbox in enumerate(bboxes_set):
#            bbox = np.array(bbox)
#            bbox = bbox.transpose()
#            if bbox.shape == (1, 3, 4):
#                bbox = np.squeeze(bbox)
#                bbox = np.expand_dims(bbox, axis = -1)
#            bbox = bbox[1]
#            b = np.zeros((len(bbox[0]), 4))
#            for j in range(4):
#                for k in range(len(bbox[j])):
#                    b[k][j] = bbox[j][k]
#            bboxes.extend(b)
#        bboxes = np.array(bboxes, dtype = np.float32)
#        
#        labels = []
#        for i, label in enumerate(labels_set):
#            label = label[1]
#            bboxes_nums.extend(label.shape)
#            labels.extend(label)
#        labels = np.array(labels, dtype = np.int64)
        
        
    bboxes = []
    #print(bboxes_set)
    for i, bbox in enumerate(bboxes_set):
        #print(bbox)
        bbox = np.array(bbox)
        #print(bbox.shape)
        bbox = bbox.transpose()
        if bbox.shape == (1, 3, 4):
            bbox = np.squeeze(bbox)
            bbox = np.expand_dims(bbox, axis = -1)
        bbox = bbox[1]
        #print(bbox)
        #print(len(bbox[0]))
        b = np.zeros((len(bbox[0]), 4))
        #print(b.shape)
        for j in range(4):
            for k in range(len(bbox[j])):
                b[k][j] = bbox[j][k]
        #print(b)
        bboxes.append(b)
        
        
    #print(bboxes[1])
    #这么一搞的话，bboxes就是一组图里所有的box的点。
    #bboxes[0]为第一张图的所有box的点
    #bboxes[0][0]位第一张图的第一个box的所有的点。[ymin, xmin, ymax, xmax]
    
    
    labels = []
    for i, label in enumerate(labels_set):
        label = np.array(label)
        label = label[1]
        labels.append(label)
        
    
    return image_set, bboxes_set, labels_set #, bboxes_nums


def resize_img_to_300(img_data, height, width, channels):
    #因为可能会导致BUG而弃用
    #这里用于显示图片,这里读取图片是用tensorflow的一种特殊的方法读的，如果要显示需要先解码
    #img_data = tf.image.decode_jpeg(img_data)
    #这里有个问题，method有0 1 2 3四种，只有选择method 1，也就是最近邻居法的时候，才可以正常显示
    #因为其他三个接受的应该是tf.float32 而不是tf.uint8
    #img_data = tf.image.resize_images(img_data, [300, 300], method = 1)
    
    img_data = tf.decode_raw(img_data, tf.uint8)
    img_data = tf.reshape(img_data, [height, width+1, channels])
    #print(img_data)
    return img_data

def get_bboxes(path, tfrecord_filename):
    """
        Look Down
    """
    _, bboxes_set, labels_set = read_from_tfrecords_tf(path, tfrecord_filename)
    
    bboxes = []   
    #print(bboxes_set)
    
    for i, bbox in enumerate(bboxes_set):
        #if i == 1: print(bbox)
        bbox = np.array(bbox)
        #print(bbox.shape)
        bbox = bbox.transpose()
        if bbox.shape == (1, 3, 4):
            bbox = np.squeeze(bbox)
            bbox = np.expand_dims(bbox, axis = -1)
        bbox = bbox[1]

        b = np.zeros((len(bbox[0]), 4))
        #print(b.shape)
        for j in range(4):
            for k in range(len(bbox[j])):
                b[k][j] = bbox[j][k]
        bboxes.extend(b)
    bboxes = np.array(bboxes, dtype = np.float32)
    print(bboxes.shape)
    #print(np.array(bboxes, dtype=np.float32))
    #这么一搞的话，bboxes就是一组图里所有的box的点。
    #bboxes[0]为第一张图的所有box的点
    #bboxes[0][0]位第一张图的第一个box的所有的点。[ymin, xmin, ymax, xmax]
    return bboxes

def get_labels(path, tfrecord_filename):
    """
        Look Down
    """
    _, bboxes_set, labels_set = read_from_tfrecords_tf(path, tfrecord_filename)
    
    labels = []
    for i, label in enumerate(labels_set):
        label = np.array(label)
        label = label[1]
        labels.append(label)
        
    #print(labels[1])
    return labels
    

def get_bboxes_and_labels(path, tfrecord_filename):
    """
        Look Down
    """
    _, bboxes_set, labels_set = read_from_tfrecords_tf(path, tfrecord_filename)
    
    bboxes = []
    for i, bbox in enumerate(bboxes_set):
        bbox = np.array(bbox)
        #print(bbox.shape)
        bbox = bbox.transpose()
        if bbox.shape == (1, 3, 4):
            bbox = np.squeeze(bbox)
            bbox = np.expand_dims(bbox, axis = -1)
        bbox = bbox[1]
        #print(bbox)
        #print(len(bbox[0]))
        b = np.zeros((len(bbox[0]), 4))
        #print(b.shape)
        for j in range(4):
            for k in range(len(bbox[j])):
                b[k][j] = bbox[j][k]
        #print(b)
        bboxes.append(b)
    #print(bboxes[1])
    #这么一搞的话，bboxes就是一组图里所有的box的点。
    #bboxes[0]为第一张图的所有box的点
    #bboxes[0][0]位第一张图的第一个box的所有的点。[ymin, xmin, ymax, xmax]
    labels = []
    for i, label in enumerate(labels_set):
        label = np.array(label)
        label = label[1]
        labels.append(label)
        
    #print(labels[1])
    return bboxes, labels

#read_from_tfrecords_tf('C:\\Users\\Stpraha\\SSD_Light\\tfrecords\\', 'voc_train_134.tfrecord')
#get_bboxes('C:\\Users\\Stpraha\\SSD_Light\\tfrecords\\', 'voc_train_134.tfrecord')

def read_batch_data(batch_num, path = './tfrecords/'):
    filenames = os.listdir(path)
    #print(len(filenames))
    batch_num = batch_num % len(filenames)
    
    image_set, bboxes_set, labels_set = read_from_tfrecords_tf(path, filenames[batch_num])
    
    images = np.array(image_set, dtype = 'float32')
    
    bboxes = []
    for i, bbox in enumerate(bboxes_set):
        bbox = np.array(bbox)
        bbox = bbox.transpose()
        if bbox.shape == (1, 3, 4):
            bbox = np.squeeze(bbox)
            bbox = np.expand_dims(bbox, axis = -1)
        bbox = bbox[1]
        b = np.zeros((len(bbox[0]), 4))
        for j in range(4):
            for k in range(len(bbox[j])):
                b[k][j] = bbox[j][k]
        bboxes.append(b)
        
    labels = []
    for i, label in enumerate(labels_set):
        label = np.array(label)
        label = label[1]
        labels.append(label)
    
    return images, bboxes, labels
    
#read_batch_data(0)



