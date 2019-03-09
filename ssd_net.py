# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 23:24:58 2018

@author: Stpraha
"""
import read_tfrecord
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

slim = tf.contrib.slim
#tf.reset_default_graph()

anchor_sizes=[(21., 45.),
              (45., 99.),
              (99., 153.),
              (153., 207.),
              (207., 261.),
              (261., 315.)]

anchor_ratios=[[2, .5],
               [2, .5, 3, 1./3],
               [2, .5, 3, 1./3],
               [2, .5, 3, 1./3],
               [2, .5],
               [2, .5]]



def l2_normalization(layer, scale, trainable = True):
    with tf.variable_scope('l2_norm', reuse = tf.AUTO_REUSE):
        n_channels = layer.get_shape().as_list()[-1]
        l2_norm = tf.nn.l2_normalize(layer, [3], epsilon = 1e-12)
        gamma = tf.get_variable("gamma", shape = [n_channels, ], dtype = tf.float32,
                                initializer = tf.constant_initializer(scale), trainable = trainable)
        layer = l2_norm * gamma
        return layer

def ssd_get_prediction(layers,
                       num_classes = 21,
                       sizes = anchor_sizes,
                       ratios = anchor_ratios):
    """
    Argument:
        layers: net4, net7 ......
        num_classes: 20 classes + 1 background class
        sizes:
        ratios:
    Return:
        predictions: for example , net 7, return batch_size x (19*19)*6*(4) 
                     the '4' is used for box
        localizations: for example , net 7, return batch_size x (19*19)*6*(20+1) 
                       the '20+1' is used for classification
        logits: similar to predictions, used for loss
    """
    with tf.variable_scope('get_pred', reuse = tf.AUTO_REUSE):
    
        predictions = []
        logits = []
        localizations = []
        
        for i, layer in enumerate(layers):
            #在VGG的第四层后面需要加一个L2Normalization
            #因为该层比较靠前，其norm较大
            if i == 0:
                layer = l2_normalization(layer, 20)
                
            #Number of anchors 
            #第VGG CONV4层，有4种anchor， 后面VGG连续三个6种anchor，再后面两个4种anchor
            num_anchor = len(sizes[i]) + len(ratios[i])
            
            #Location
            num_loc_pred = num_anchor * 4  #每种anchor都对应4个参数鸭
            loc_pred = slim.conv2d(layer, num_loc_pred, [3, 3], activation_fn = None)
            #这里，num_loc_pred = num_anchor * 4, 所以能正好reshape完。
            loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [num_anchor, 4])  
            #print(loc_pred.shape.as_list(), 'sdfasd')
           
            #Class prediction
            num_cls_pred = num_anchor * num_classes
            cls_pred = slim.conv2d(layer, num_cls_pred, [3, 3], activation_fn = None)
            cls_pred = tf.reshape(cls_pred, cls_pred.get_shape().as_list()[: -1] + [num_anchor, num_classes])
            #print(cls_pred.shape.as_list(), 'cls_pred  shape')
            cls_pred_final = slim.softmax(loc_pred)
            
            logits.append(cls_pred)
            localizations.append(loc_pred)
            predictions.append(cls_pred_final)
            
        #print(predictions)
        #print(logits)
        #print(localizations)
        return predictions, logits, localizations
    
    
def ssd_get_prediction_single(layer,
                              i,
                              num_classes = 21,
                              sizes = anchor_sizes,
                              ratios = anchor_ratios):
    #在VGG的第四层后面需要加一个L2Normalization
    #因为该层比较靠前，其norm较大
    if i == 0:
        layer = l2_normalization(layer, 20)
                
    #Number of anchors 
    #第VGG CONV4层，有4种anchor， 后面VGG连续三个6种anchor，再后面两个4种anchor
    num_anchor = len(sizes[i]) + len(ratios[i])
            
    #Location
    num_loc_pred = num_anchor * 4  #每种anchor都对应4个参数鸭
    loc_pred = slim.conv2d(layer, num_loc_pred, [3, 3], activation_fn = None)
    #这里，num_loc_pred = num_anchor * 4, 所以能正好reshape完。
    loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [num_anchor, 4])  
    print(loc_pred.shape.as_list(), 'sdfasd')
    loc_pred_final = slim.softmax(loc_pred)
    print(loc_pred_final.get_shape().as_list())
                       
    #Class prediction
    num_cls_pred = num_anchor * num_classes
    cls_pred = slim.conv2d(layer, num_cls_pred, [3, 3], activation_fn = None)
    cls_pred = tf.reshape(cls_pred, cls_pred.get_shape().as_list()[: -1] + [num_anchor, num_classes])
    print(cls_pred.shape.as_list(), 'cls_pred  shape')
            
    #logits.append(loc_pred)
    #predictions.append(loc_pred_final)
    #localizations.append(cls_pred)
            
    #print(predictions.get_shape().as_list(), localizations.get_shape().as_list())
    #return predictions, logits, localizations        

def ssd_net(inputs,
            dropout_keep_prob = 0.7,
            is_training = True,
            num_classes = 21,
            sizes = anchor_sizes,
            ratios = anchor_ratios):
    
    with tf.variable_scope('ssd_net', reuse = False):
        #Original VGG-16 blocks.
        #slim.repeat 即为执行多次，并会智能地更改scope的下标
        layer1 = tf.layers.conv2d(inputs, 128, 3, strides = 2, padding = 'same')
        
        #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope = 'conv1')
        #net = slim.max_pool2d(net, [2, 2], scope = 'pool1')
        net = slim.conv2d(inputs, 64, [3, 3], scope = 'conv1_1', padding = 'SAME')
        net = slim.conv2d(net, 64, [3, 3], scope = 'conv1_2', padding = 'SAME')
        net = slim.max_pool2d(net, [2, 2], scope = 'pool1', padding = 'SAME')
        net1 = net
        #Block2
        net = slim.conv2d(net, 128, [3, 3], scope = 'conv2_1', padding = 'SAME')
        net = slim.conv2d(net, 128, [3, 3], scope = 'conv2_2', padding = 'SAME')
        net = slim.max_pool2d(net, [2, 2], scope = 'pool2', padding = 'SAME')
        net2 = net
        #Block3
        net = slim.conv2d(net, 256, [3, 3], scope = 'conv3_1', padding = 'SAME')
        net = slim.conv2d(net, 256, [3, 3], scope = 'conv3_2', padding = 'SAME')
        net = slim.conv2d(net, 256, [3, 3], scope = 'conv3_3', padding = 'SAME')
        net = slim.max_pool2d(net, [2, 2], scope = 'pool3', padding = 'SAME')
        net3 = net
        #Block4
        net = slim.conv2d(net, 512, [3, 3], scope = 'conv4_1', padding = 'SAME')
        net = slim.conv2d(net, 512, [3, 3], scope = 'conv4_2', padding = 'SAME')
        net = slim.conv2d(net, 512, [3, 3], scope = 'conv4_3', padding = 'SAME')
        net = slim.max_pool2d(net, [2, 2], stride = 1, scope = 'pool4', padding = 'SAME')
        net4 = net
        #Block5
        net = slim.conv2d(net, 512, [3, 3], scope = 'conv5_1', padding = 'SAME')
        net = slim.conv2d(net, 512, [3, 3], scope = 'conv5_2', padding = 'SAME')
        net = slim.conv2d(net, 512, [3, 3], scope = 'conv5_3', padding = 'SAME')
        net = slim.max_pool2d(net, [3, 3], scope = 'pool5', padding = 'SAME')    
        net5 = net
     
        #Additional SSD blocks.
        #Two fully connected layers
        #Block 6:    #rate: 对于使用atrous convolution的膨胀率
        net = slim.conv2d(net, 1024, [3, 3], scope = 'conv6', padding = 'SAME')
        net = tf.layers.dropout(net, rate = dropout_keep_prob, training = is_training)
        net6 = net
        #Block 7:
        net = slim.conv2d(net, 1024, [1, 1], scope = 'conv7', padding = 'SAME')
        net = tf.layers.dropout(net, rate = dropout_keep_prob, training = is_training)
        net7 = net
        
        #Additional four layers
        #Block 8:
        net = slim.conv2d(net7, 256, [1, 1], scope = 'conv8_1', padding = 'SAME')
        net = slim.conv2d(net, 512, [3, 3], stride = 2, scope = 'conv8_2', padding = 'SAME')
        net8 = net
        #Block 9:
        net = slim.conv2d(net8, 128, [1, 1], scope = 'conv9_1', padding = 'SAME')
        net = slim.conv2d(net, 256, [3, 3], stride = 2, scope = 'conv9_2', padding = 'SAME')
        net9 = net
        #Block 10:
        net = slim.conv2d(net9, 128, [1, 1], scope = 'conv10_1', padding = 'SAME')
        net = slim.conv2d(net, 256, [3, 3], stride = 2, scope = 'conv10_2', padding = 'SAME')
        net10 = net
        #Block 11:
        net = slim.conv2d(net10, 128, [1, 1], scope = 'conv11_1', padding = 'SAME')
        net = slim.conv2d(net, 256, [3, 3], stride = 2, scope = 'conv11_2', padding = 'VALID')
        net11 = net
           
        predictions = []
        logits = []
        localizations = []
        
        layers = [net4, net7, net8, net9, net10, net11]
        
        for i, layer in enumerate(layers):
            #在VGG的第四层后面需要加一个L2Normalization
            #因为该层比较靠前，其norm较大
            if i == 0:
                layer = l2_normalization(layer, 20)
                
            #Number of anchors 
            #第VGG CONV4层，有4种anchor， 后面VGG连续三个6种anchor，再后面两个4种anchor
            num_anchor = len(sizes[i]) + len(ratios[i])
            
            #Location
            num_loc_pred = num_anchor * 4  #每种anchor都对应4个参数鸭
            loc_pred = slim.conv2d(layer, num_loc_pred, [3, 3], activation_fn = None)
            #这里，num_loc_pred = num_anchor * 4, 所以能正好reshape完。
            loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [num_anchor, 4])  
            #print(loc_pred.shape.as_list(), 'sdfasd')
           
            #Class prediction
            num_cls_pred = num_anchor * num_classes
            cls_pred = slim.conv2d(layer, num_cls_pred, [3, 3], activation_fn = None)
            cls_pred = tf.reshape(cls_pred, cls_pred.get_shape().as_list()[: -1] + [num_anchor, num_classes])
            #print(cls_pred.shape.as_list(), 'cls_pred  shape')
            cls_pred_final = slim.softmax(cls_pred)
            
            logits.append(cls_pred)
            localizations.append(loc_pred)
            predictions.append(cls_pred_final)
            
#            print('cls_pred_final',cls_pred_final.shape)
#            print('local_pred', loc_pred.shape)
#            print('cls_pred', cls_pred.shape)
    #return net4, net7, net8, net9, net10, net11
    #print(predictions.shape)
    
    return predictions, logits, localizations

    

def test_ssd_net():
    img_features, _, _ = read_tfrecord.read_from_tfrecords_tf('C:\\Users\\Stpraha\\SSD_Light\\tfrecords\\', 'voc_train_134.tfrecord')
    net4, net7, net8, net9, net10, net11 = ssd_net(img_features)
    layers = [net4, net7, net8, net9, net10, net11]
    
    #for i, layer in enumerate(layers):
        #ssd_get_prediction_single(layer, i)
    predictions, logits, localizations = ssd_get_prediction(layers)
    print(logits)

    return predictions, logits, localizations





    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    