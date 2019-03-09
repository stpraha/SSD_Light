# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:26:47 2018

@author: Stpraha
"""
import tensorflow as tf
import ssd_net
slim = tf.contrib.slim
 
def smooth_l1_loss(gt_localizations, pred_localizations):
    """
        Smooth L1 loss
    """   
    gt_localizations = tf.cast(gt_localizations, tf.float32)
    x = tf.subtract(gt_localizations, pred_localizations)
    abs_x = tf.abs(x)
    l1 = 0.5 * (x**2.0)
    l2 = abs_x - 0.5
    
    condition = tf.less(abs_x, 1.0)
    r = tf.where(condition, l2, l1)
    
    return r


def ssd_losses(img_feature, 
               concat_glabels, 
               concat_gscores, 
               concat_glocalizations, 
               batch_size,
               threshold = 0.5,  #阈值
               alpha = 1,    #d
               negative_ratio = 2,  #正负样本size比
               num_classes = 21
               ):
    """
    The overall objective loss function is a weighted sum of the localiazation loss (loc) and the confidence loss(conf)
        L(x, c, l, g) = 1/N x ((L_conf(x,c) + alpha x L_loc(x, l, g))
    where N is the number of matched default BOXES. if N == 0, set loss --> 0
    what x means see the paper.
    localization loss: Smooth L1 loss. Between the prediction box (l) and the GT box (g)

    confidence loss: Softmax loss over multiple classes confidences (c).
    
    """
    predictions, logits, localizations = ssd_net.ssd_net(img_feature)
    
    with tf.name_scope('ssd_losses'):
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

        #这里找出score高于阈值的那些，并把它们变成1，如果低于阈值，则变成0
        #pmask 正样本mask
        pmask = concat_gscores > threshold      #return a tensor, if True, parameter is True, threshold IOU
        fpmask = tf.cast(pmask, tf.float32)     #boolean --> float
        
        num_positive = tf.reduce_sum(fpmask)    #the number of positive sample
  
        #hard negative mining  即找一些不容易划分的负样本，来增强判别性能
        no_classes = tf.cast(pmask, tf.int32)
        predictions = tf.nn.softmax(concat_logits)  #8732 * 21
        
        nmask = tf.logical_not(pmask)   #the one not positive is negative
        fnmask = tf.cast(nmask, tf.float32)
        #predictions[:,0]即为取第一列...predictions为87320 x 21 的二维矩阵
        #再说一遍tf.where，nmask为condition，如果condition为true，不变，false，变为1. - fnmask
        part_predictions = tf.expand_dims(predictions[:, 0], axis = 1)
        #nmask和fnmask是对应的，如果nmask里的是false，也就是说，是正样本，就把part_predictions里的那部分变成1，否则就不变
        #predictions里原来的是什么呢？是21个类的得分，第一列不出意外的话，是背景的可能性。
        #也就是说，如果预测的是正样本，就把这个分数变成1，如果不是正样本，分数就不变
        nvalues = tf.where(nmask, part_predictions, 1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        
        #然后再来看看选多少个负样本，因为负样本多了的话样本就倾斜了，样本比例有negative_rnum_classatio决定
        #负样本最多有多少个？
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        #num_negative 为有样本比 * 正样本个数  ？原版这里为什么还假了batch_size？
        num_negative = tf.cast(negative_ratio * num_positive, tf.int32)
        #当然负样本的个数不能比总负样本个数多啊
        num_negative = tf.minimum(num_negative, max_neg_entries)
        #先把nvalues_flat取反, 这样1就变成了-1，越接近0越大
        #也就是取了预测中最接近0的 ，个数为num_negative的负样本。
        #说是用了Hard negative mining，无法理解
        val, idxes = tf.nn.top_k(-nvalues_flat, k = num_negative)
        #啥意思？取了val最底下的哪一行，然后再取反？也就是负样本的最高分？
        max_hard_pred = -val[-1]
        #然后与运算？如果负样本的分数比最大值还高，就把它当成正样本了，是不是多此一举？
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, tf.float32)
        
        #confidience loss
        #cross_entropy       
        with tf.name_scope('cross_entropy_positive'):    
            concat_glabels = tf.reshape(concat_glabels, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = concat_logits, labels = concat_glabels)
            loss = tf.reduce_sum(loss * fpmask)
            loss = tf.div(loss, batch_size)
            tf.losses.add_loss(loss)
            
        with tf.name_scope('cross_entropy_negative'):
            no_classes = tf.reshape(no_classes, [-1])
            #这个no classes 啥意思？
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = concat_logits, labels = no_classes)
            loss = tf.reduce_sum(loss * fnmask)
            loss = tf.div(loss, batch_size)
            tf.losses.add_loss(loss)
        
        #Localizations loss
        #Smooth L1
        with tf.name_scope('localizations_loss'):
            #这个weights派什么用场？loss x weights之后是87320 x 87320 x 4
            weights = alpha * fpmask
            loss = smooth_l1_loss(concat_glocalizations, concat_localizations)
            loss = tf.reduce_sum(loss * weights)
            loss = tf.divide(loss, batch_size)
            tf.losses.add_loss(loss)
        
    return loss
    
    

   


        
        
        





