# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 22:45:59 2018
@author: Stpraha
"""
import numpy as np
import tensorflow as tf
import generate_anchor
import read_pic


#all_anchors = generate_anchor.generate_anchor()
#num_classes = 21

#print(len(all_anchors))

def flatten(
            glabels,    #gt类别
            glocalizations,  #gt位置
            gscore,  #gt分数):
            ):
    #logits_shape = logits[0].shape
    #num_classes = logits_shape[4]
        
    flatten_glabels = []
    flatten_gscores = []
    flatten_glocalizations = []
        
    #The length of logits --list-- is 6. for each layer, a logit is generated
    for i in range(6):
        # batch_size x h x w x achor_sizes x num_classes ---> [-1] x num_classes
        #flatten_logits.append(tf.reshape(logits[i], [-1, num_classes]))
        # batch_size x h x w x anchor_size ---> [-1] x 1
        flatten_glabels.append(tf.reshape(glabels[i], [-1, 1]))
        # batch_size x h x w x anchor_size x 4(x,y,h,w) ---> [-1] x 4
        flatten_gscores.append(tf.reshape(gscore[i], [-1, 1]))
        # batch_size x h x w x anchor_size x 4(x,y,h,w) ---> [-1] x 4
        #flatten_localizations.append(tf.reshape(localizations[i], [-1, 4]))
        # batch_size x h x w x anchor_size ---> [-1] x 1
        flatten_glocalizations.append(tf.reshape(glocalizations[i], [-1, 4]))
        #1st dims of the above 4 lists are same.
    
    #make 6 flatten_logits[] together
    #concat_logits = tf.concat(flatten_logits, axis = 0)
    concat_glabels = tf.concat(flatten_glabels, axis = 0)
    concat_gscores = tf.concat(flatten_gscores, axis = 0)
    #concat_localizations = tf.concat(flatten_localizations, axis = 0)
    concat_glocalizations = tf.concat(flatten_glocalizations, axis = 0)

    return concat_glabels, concat_gscores, concat_glocalizations


def layer_anchor_encode(layer_anchors):
    
    #先得到anchor的坐标和面积（？）

    yref, xref, href, wref = layer_anchors
    #这里将href降维，以便做减法。。早知道在上一级里就不增加维度了。。
    href = tf.squeeze(href)
    wref = tf.squeeze(wref)
  
    ymin = yref - href / 2.  #上边界
    xmin = xref - wref / 2.  #左边界
    ymax = yref + href / 2.  #下边界
    xmax = xref + wref / 2.  #右边界
     
    #get_vol
    anchor_vol = (xmax - xmin) * (ymax - ymin)

    #计算shape？有必要么？
    #print(anchor_vol.shape)
    #print(yref.shape[0], yref.shape[1], href.shape[0])
   
    return ymin, xmin, ymax, xmax, anchor_vol
    
def jaccard_with_anchors(anchor_vol, ymin, xmin, ymax, xmax, bbox):
    """
         Jaccard score: J(A, B) = (A inter B) / (A union B)
    """   
    #Intersection 即计算anchor和box之间的交集
    #这里求出交集方块的四个边界
    inter_ymin = tf.maximum(ymin, bbox[0])
    inter_xmin = tf.maximum(xmin, bbox[1])
    inter_ymax = tf.minimum(ymax, bbox[2])
    inter_xmax = tf.minimum(xmax, bbox[3])
    #这里求出交集方块的面积
    h = tf.maximum(inter_ymax - inter_ymin, 0.)
    w = tf.maximum(inter_xmax - inter_xmin, 0.)
    inter_vol = h * w
    #print(inter_vol)
    #Union 即计算anchor和box之间的并集
    #A union B = A + B - (A inter B)
    #Single bbox: [ymin, xmin, ymax, xmax]
    bbox_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    union_vol = anchor_vol + bbox_vol - inter_vol
    
    #Finally, inter / union
    #在(38, 38)时候这应该是一个较为稀疏的矩阵（划掉）Tensor
    jaccard = tf.div(inter_vol, union_vol)
    
    #with tf.Session() as sess:
        #print(sess.run(jaccard))
        
    return jaccard

def one_box_one_layer_process(layer, label, box, layer_index, prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    #bboxes = read_tfrecord.get_bboxes('C:\\Users\\Stpraha\\SSD_Light\\tfrecords\\', 'voc_train_134.tfrecord')   
    #print(bboxes[0][0])
    #labels = read_tfrecord.get_labels('C:\\Users\\Stpraha\\SSD_Light\\tfrecords\\', 'voc_train_134.tfrecord')         
    #above OK
    #ymin, xmin, ymax, xmax, anchor_vol = bboxes_encode_layer()
    #jaccard = jaccard_with_anchors(anchor_vol, ymin, xmin, ymax, xmax, bboxes[0][0])
    
    #for i, layer in enumerate(anchor_layers):
    #print(len(layer))
    yref, xref, href, wref = layer
    #这里将href降维，以便做减法。。早知道在上一级里就不增加维度了。。
    #print(href)
    href = tf.squeeze(href)
    wref = tf.squeeze(wref)  
    
    ymin, xmin, ymax, xmax, anchor_vol = layer_anchor_encode(layer)   
    
        
    with tf.Session() as sess:
        if layer_index == 2:
            print(sess.run(href))
    
    shape = anchor_vol.shape
    #print(shape)
    feature_labels = tf.zeros(shape, dtype = tf.int64)
    feature_scores = tf.zeros(shape, dtype = tf.float64)
    feature_ymin = tf.zeros(shape, dtype = tf.float64)
    feature_xmin = tf.zeros(shape, dtype = tf.float64)
    feature_ymax = tf.ones(shape, dtype = tf.float64)
    feature_xmax = tf.ones(shape, dtype = tf.float64)
        
    #对每个box    
    #print('len(box)', len(box))
    for j in range(len(box)):
    #for j in range(1):
        #print(len(bboxes[i]))
        single_label = label[j]
        single_box = box[j]
        jaccard = jaccard_with_anchors(anchor_vol, ymin, xmin, ymax, xmax, single_box)

        mask = tf.greater(jaccard, feature_scores)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, tf.float64)
        
        feature_labels = imask * single_label + (1 - imask) * feature_labels
        #tf.where(input, a,b)，其中a，b均为尺寸一致的tensor，
        #作用是将a中对应input中true的位置的元素值不变，其余元素进行替换，替换成b中对应位置的元素值
        feature_scores = tf.where(mask, jaccard, feature_scores)
#        with tf.Session() as sess:
#            print(sess.run(feature_scores))
        feature_ymin = fmask * single_box[0] + (1 - fmask) * feature_ymin
        feature_xmin = fmask * single_box[1] + (1 - fmask) * feature_xmin
        feature_ymax = fmask * single_box[2] + (1 - fmask) * feature_ymax
        feature_xmax = fmask * single_box[3] + (1 - fmask) * feature_xmax
    
    #计算补偿后的中心
    feature_cy = (feature_ymax + feature_ymin) / 2
    feature_cx = (feature_xmax + feature_xmin) / 2
    #计算补偿后的宽和高
    feature_h = (feature_ymax - feature_ymin) / 2
    feature_w = (feature_xmax - feature_xmin) / 2
    
    #Encode features
    #print((feature_cy - yref) / href)
    feature_cy = (feature_cy - yref) / href# / prior_scaling[0]
    feature_cx = (feature_cx - xref) / wref / prior_scaling[1]
    feature_h = tf.log(feature_h / href) / prior_scaling[2]
    feature_w = tf.log(feature_w / wref) / prior_scaling[3]
#    
#    with tf.Session() as sess:
#        print(sess.run(feature_cy))
    
    feature_localizations = tf.stack([feature_ymin, feature_xmin, feature_ymax, feature_xmax])
    feature_localizations = tf.transpose(feature_localizations, perm = [1, 2, 3, 0])
    
    #print(layer)
    #print(layer_index, feature_localizations)

    
    return feature_localizations, feature_labels, feature_scores
    
    
#上面的，确定的图片和layer
#下面从单一图片的6个layer-->单一layer的batchsize图片
    

#图片层面
#one_box_one_layer_process(layer, label, box, prior_scaling=[0.1, 0.1, 0.2, 0.2]):
def all_bboxes_labels_process(layer, bboxes, labels, layer_i):
    batch_single_layer_localizations = []
    batch_single_layer_labels = []
    batch_single_layer_scores = []

    for i in range(len(bboxes)):
        processed_localizations, processed_labels, processed_scores = one_box_one_layer_process(layer, labels[i], bboxes[i], layer_i)

        batch_single_layer_localizations.append(processed_localizations) 
        batch_single_layer_labels.append(processed_labels)
        batch_single_layer_scores.append(processed_scores)
    
    batch_single_layer_localizations = tf.convert_to_tensor(batch_single_layer_localizations)    
    batch_single_layer_labels = tf.convert_to_tensor(batch_single_layer_labels)    
    batch_single_layer_scores = tf.convert_to_tensor(batch_single_layer_scores)    
    #print(batch_single_layer_localizations)
    return batch_single_layer_localizations, batch_single_layer_labels, batch_single_layer_scores
 

#layer层面       
def all_layers_process(bboxes, labels, anchor_layers):
    #print('get all_anchor_layers finished !')
    ########这里有错
    all_localizations = []
    all_labels = []
    all_scores = []
    for i, layer in enumerate(anchor_layers):
        processed_localizations, processed_labels, processed_scores = all_bboxes_labels_process(layer, bboxes, labels, i)
        #print(processed_localizations.shape)
        all_localizations.append(processed_localizations)
        all_labels.append(processed_labels)
        all_scores.append(processed_scores)
        
    #with tf.Session() as sess:
    #print(all_localizations)
    return all_localizations, all_labels, all_scores

def test():
    img, bboxes, labels = read_pic.read_pic_batch(1, 0)
    anchor_layers = generate_anchor.generate_anchor()
    raw_glocalizations, raw_glabels, raw_gscores = all_layers_process(bboxes, labels, anchor_layers)
    
    concat_glabels, concat_gscores, concat_glocalizations = flatten(raw_glabels, raw_glocalizations, raw_gscores)
    
    with tf.Session() as sess:
        concat_glabels = sess.run(concat_glabels) 
        concat_gscores = sess.run(concat_gscores) 
        concat_glocalizations = sess.run(concat_glocalizations)
    
    print(concat_glocalizations.shape, concat_glocalizations.dtype)
    print(concat_gscores.shape)
    print(concat_glabels.shape)
    
    
   
test()