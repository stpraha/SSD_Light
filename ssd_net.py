# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 23:24:58 2018

@author: Stpraha
"""
import tensorflow as tf
slim = tf.contrib.slim

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


def pad2d(inputs, pad = (0, 0), mode = 'CONSTANT', trainable = True, scope = None):
    with tf.name_scope(scope, 'pad2d', [inputs]):
        paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        net = tf.pad(inputs, paddings, mode = mode)
        return net


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
    
    

def get_pred_loc_cls(feature_map, i, scope,  cls_num = 2):
    
    num_anchor = len(anchor_sizes[i]) + len(anchor_ratios[i])
    num_loc_pred = num_anchor * 4
    num_cls_pred = num_anchor * cls_num

    with tf.variable_scope('ssd_pred' + scope):
        net_cls = slim.conv2d(feature_map, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
        #net_cls = tf.contrib.layers.flatten(net_cls)
        net_cls = tf.reshape(net_cls, net_cls.get_shape().as_list()[:-1] + [num_anchor, cls_num])
        
        net_loc = slim.conv2d(feature_map, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
        #net_loc = tf.contrib.layers.flatten(net_loc)
        net_loc = tf.reshape(net_loc, net_loc.get_shape().as_list()[:-1] + [num_anchor, 4])
        
        #print(net_cls.shape)
        #print(net_loc.shape)
        return net_cls, net_loc
       

def ssd_net(inputs,
            dropout_keep_prob = 0.7,
            is_training = True,
            num_classes = 2,
            sizes = anchor_sizes,
            ratios = anchor_ratios,
            scope = 'ssd_net'):
    
    with tf.variable_scope(scope, [inputs], reuse = tf.AUTO_REUSE):
        #Original VGG-16 blocks.
        #slim.repeat 即为执行多次，并会智能地更改scope的下标
        #print(inputs.shape)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope = 'conv1')
        net1 = net
        net = slim.max_pool2d(net, [2, 2], scope = 'pool1')
        #Block2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope = 'conv2')
        net2 = net
        net = slim.max_pool2d(net, [2, 2], scope = 'pool2')
        #Block3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope = 'conv3')
        net3 = net
        net = slim.max_pool2d(net, [2, 2], scope = 'pool3', padding = 'SAME')
        #Block4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope = 'conv4')
        net4 = net
        #Block5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope = 'conv5')
        net5 = net
        net = slim.max_pool2d(net, [3, 3], scope = 'pool5', padding = 'SAME')    
     
        #Additional SSD blocks.
        #Two fully connected layers
        #Block 6:    #rate: 对于使用atrous convolution的膨胀率
        net = slim.conv2d(net, 1024, [3, 3], scope = 'conv6')
        net6 = net
        net = tf.layers.dropout(net, rate = dropout_keep_prob, training = is_training)

        #Block 7:
        net = slim.conv2d(net, 1024, [1, 1], scope = 'conv7')
        net7 = net
        net = tf.layers.dropout(net, rate = dropout_keep_prob, training = is_training)

        #Additional four layers
        #Block 8:
        net = slim.conv2d(net7, 256, [1, 1], scope = 'conv8_1')
        net = slim.conv2d(net, 512, [3, 3], stride = 2, scope = 'conv8_2')
        net8 = net
        
        #Block 9:
        net = slim.conv2d(net8, 128, [1, 1], scope = 'conv9_1')
        net = slim.conv2d(net, 256, [3, 3], stride = 2, scope = 'conv9_2')
        net9 = net
        
        #Block 10:
        net = slim.conv2d(net9, 128, [1, 1], scope = 'conv10_1', padding = 'VALID')
        net = slim.conv2d(net, 256, [3, 3], stride = 1, scope = 'conv10_2', padding = 'VALID')
        net10 = net
        
        #Block 11:
        net = slim.conv2d(net10, 128, [1, 1], scope = 'conv11_1', padding = 'VALID')
        net = slim.conv2d(net, 256, [3, 3], stride = 1, scope = 'conv11_2', padding = 'VALID')
        net11 = net
        
        predictions = []
        logits = []
        localizations = []
        
        layers = [net4, net7, net8, net9, net10, net11]

        preds_result = []
        preds_loc = []
        preds_cls = []
        preds_softlogit = []
        for i, layer in enumerate(layers):
            if i == 0:
                layer = l2_normalization(layer, 20)
                
            pred_cls, pred_loc = get_pred_loc_cls(layer, i, scope = scope + 'conv_' + str(i))
            #print(pred_cls.shape)
            pred_softlogit = tf.nn.softmax(pred_cls)
            pred_result = tf.arg_max(tf.nn.softmax(pred_cls), 4)
            preds_cls.append(pred_cls)
            preds_loc.append(pred_loc)
            preds_result.append(pred_result)
            preds_softlogit.append(pred_softlogit)
        #print(preds_cls)
    return preds_result, preds_cls, preds_loc, preds_softlogit

    

def test_ssd_net():
    img_features = tf.placeholder(tf.float32, [32, 300, 300, 3])
    #print(img_features.shape)
    predictions, logits, localizations = ssd_net(img_features)

    return predictions, logits, localizations

#test_ssd_net()



    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    