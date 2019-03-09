# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:10:19 2019

@author: Stpraha
"""
import tensorflow as tf
import numpy as np
import ssd_net
import generate_anchor
import loss_function
import process_ground_truth
import read_tfrecord
import read_pic

slim = tf.contrib.slim

learning_rate = 0.001
batch_size = 10


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

def get_input(batch_size):
    glabels = tf.placeholder(tf.int32, [batch_size * 8732, 1])   
    gscores = tf.placeholder(tf.float32, [batch_size * 8732, 1])
    glocalizations = tf.placeholder(tf.float32, [batch_size * 8732, 4]) 
    img_features = tf.placeholder(tf.float32, [batch_size, 300, 300, 3])
    
    return glabels, gscores, glocalizations, img_features


def get_loss(concat_glabels, concat_gscores, concat_glocalizations, concat_localizations, concat_logits, batch_size):
    loss = loss_function.ssd_losses(concat_glabels, concat_gscores, concat_glocalizations, concat_localizations, concat_logits, batch_size = batch_size)
    
    return loss
    
def get_optimizer(learning_rate, loss):
    train_vars = tf.global_variables()
    #print(train_vars)
    loss_vars = [var for var in train_vars if var.name.startswith('ssd_net')]
    #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=train_vars)
    
    return opt

def train(): 
    print('now satart training')
    img, bboxes, labels = read_tfrecord.read_batch_data(0, 'C:\\Users\\Stpraha\\SSD_Light\\tfrecords\\')
    print('welcome')
    glabels, gscores, glocalizations, img_features = get_input(batch_size)
    net4, net7, net8, net9, net10, net11 = ssd_net.ssd_net(img_features)
    layers = [net4, net7, net8, net9, net10, net11]
    anchor_layers = generate_anchor.generate_anchor()
    predictions, logits, localizations = ssd_net.ssd_get_prediction(layers)
    raw_glocalizations, raw_glabels, raw_gscores = process_ground_truth.all_layers_process(bboxes, labels, anchor_layers)
    concat_glabels, concat_gscores, concat_glocalizations = flatten(raw_glabels, raw_glocalizations, raw_gscores)
    loss = loss_function.ssd_losses(glabels, gscores, glocalizations, logits, localizations, batch_size = batch_size)
    opt = get_optimizer(learning_rate, loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('asdfasdfasdf')
        epoches = 10
        for i in range(epoches):
            print('hahahah here')
            iter_loss = 0
            batch_nums = 0
            for j in range(78):
                img, bboxes, labels = read_tfrecord.read_batch_data(0, 'C:\\Users\\Stpraha\\SSD_Light\\tfrecords\\')    
                raw_glocalizations, raw_glabels, raw_gscores = process_ground_truth.all_layers_process(bboxes, labels, anchor_layers)
                concat_glabels, concat_gscores, concat_glocalizations= flatten(raw_glabels, raw_glocalizations, raw_gscores)
                
                concat_glabels = sess.run(concat_glabels) 
                concat_gscores = sess.run(concat_gscores) 
                concat_glocalizations = sess.run(concat_glocalizations) 
            
                #print(concat_logits)
                _, out_loss = sess.run([opt, loss], feed_dict = {img_features : img,
                                                             glabels : concat_glabels, 
                                                             gscores : concat_gscores, 
                                                             glocalizations : concat_glocalizations})
                
                print('now it is running!')
                print(out_loss)
                #print(localizations)

    
#with tf.Graph().as_default():
#train()
    