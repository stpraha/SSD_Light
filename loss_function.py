# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:26:47 2018

@author: Stpraha
"""
import tensorflow as tf
import ssd_net
import numpy as np
slim = tf.contrib.slim
 
def modified_smooth_l1_loss(bbox_pred, bbox_targets, scope, bbox_inside_weights = 1., bbox_outside_weights = 1., sigma = 1):
    """
        result = outside_weights * Smooth_L1_loss(inside_weights * (pred_loc - gt_loc))
        Smooth_L1_loss(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                            |x| - 0.5 / sigma^2,    otherwise
    """
    with tf.name_scope(scope, [bbox_pred, bbox_targets]):
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

       
    return outside_mul

def smooth_l1_loss(gt_localizations, pred_localizations):
    """
        Smooth L1 loss
    """
    gt_localizations = tf.cast(gt_localizations, tf.float32)
    x = tf.subtract(gt_localizations, pred_localizations)
    abs_x = tf.abs(x)
    l1 = 0.5 * (abs_x**2.0)
    l2 = abs_x - 0.5
    
    condition = tf.less(abs_x, 1.0)
    result = tf.where(condition, l2, l1)
    
    return result

def ssd_losses(img_feature, 
               concat_glabels, 
               concat_gscores, 
               concat_glocalizations, 
               batch_size,
               threshold = 0.5,  #阈值
               alpha = 1,    #d
               negative_ratio = 2,  #正负样本size比
               num_classes = 2,
               scope = 'ssd_losses'
               ):
    """
    The overall objective loss function is a weighted sum of the localiazation loss (loc) and the confidence loss(conf)
        L(x, c, l, g) = 1/N x ((L_conf(x,c) + alpha x L_loc(x, l, g))
    where N is the number of matched default BOXES. if N == 0, set loss --> 0
    what x means see the paper.
    localization loss: Smooth L1 loss. Between the prediction box (l) and the GT box (g)

    confidence loss: Softmax loss over multiple classes confidences (c).
    
    """
    predictions, logits, localizations, softlogits = ssd_net.ssd_net(img_feature)
    #print(concat_glabels)
    with tf.name_scope(scope):
        #-----------------------------------------------------------------------------------------------------
        #Notice: logits: list of array. 6 x batch_size x may(38x38x4x21)
        #   6 is the layer amount, also the length of list.
        #   The array, membership of the list, Ex. batch_sizex38x38x4x21.
        #   However, the concat_... are in the same arrangement. Look process_ground_truth.py for details.
        #   So we just follow this way.
        #------------------------------------------------------------------------------------------------------
        flatten_logits = []
        flatten_localizations = []
        for i in range(len(logits)):
            flatten_layer_logits = []
            flatten_layer_localizations = []
            for j in range(batch_size):
                flatten_layer_logits.append(tf.reshape(logits[i][j], [-1, num_classes]))
                flatten_layer_localizations.append(tf.reshape(localizations[i][j], [-1, 4]))
            #make batch flatten_pic_logits, flatten_pic_localizations together
            flatten_layer_logits = tf.concat(flatten_layer_logits, axis = 0)
            flatten_layer_localizations = tf.concat(flatten_layer_localizations, axis = 0)

            flatten_logits.append(flatten_layer_logits)
            flatten_localizations.append(flatten_layer_localizations)
            
        #make layer flatten_pic_logits, flatten_pic_localizations together
        #concat_logits --> [batchsize*8732, 21]
        #concat_localizations -->[batch_size*8732, 4]
        concat_logits = tf.concat(flatten_logits, axis = 0)
        concat_localizations = tf.concat(flatten_localizations, axis = 0)
            
        #confidience loss
        #cross_entropy       
        
        print('asdfasdfas', concat_logits.shape)
        concat_glabels = tf.reduce_sum(concat_glabels, axis = 1)
        print('asdfas', concat_glabels.shape)

        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = concat_glabels, logits = concat_logits)
        cls_loss = tf.reduce_sum(cross_entropy)
        
        loc_loss = smooth_l1_loss(concat_glocalizations[-30:-1], concat_localizations[-30:-1])
        loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis = -1))

        total_loss = tf.add(cls_loss, loc_loss, name = 'total_loss')
        
        
    return total_loss
    
    

   


        
        
        





