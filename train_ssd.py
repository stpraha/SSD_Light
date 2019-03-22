# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:10:19 2019

@author: Stpraha
"""
import tensorflow as tf
import loss_function
import process_ground_truth
import read_pic
import os
import encode_predictions
import ssd_net
import numpy as np
import draw_pic
import cv2
slim = tf.contrib.slim

sample_nums = 273
positive_threshold = 0.5
class_num = 4
negative_ratio = 2  #not used yet
loss_positive_threshold = 0.5   #not used yet


def flatten(glabels, glocalizations, gscore):
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
    img_feature = tf.placeholder(tf.float32, [batch_size, 300, 300, 3])
    
    return glabels, gscores, glocalizations, img_feature

def get_loss(img_feature, glabels, gscores, glocalizations, batch_size):
    loss = loss_function.ssd_losses(img_feature, 
                                    glabels, 
                                    gscores, 
                                    glocalizations, 
                                    batch_size = batch_size, 
                                    threshold = loss_positive_threshold,
                                    negative_ratio = negative_ratio, 
                                    num_classes = class_num
                                   )
    
    return loss
    
def get_optimizer(learning_rate, loss):
    train_vars = tf.global_variables()
    #print(train_vars)
    ssd_vars = [var for var in train_vars if var.name.startswith('ssd_net')]
    #print(ssd_vars)
    #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = ssd_vars)
    
    return opt


def draw(img, boxes):
    img_file_name = './out/' + '0090' + '_ssd.jpg'
    
    for i in range(boxes.shape[0]):
        box = boxes[i]
        ymin = int(np.maximum(0, box[0] * 300))
        xmin = int(np.maximum(0, box[1] * 300))
        ymax = int(np.minimum(300, box[2] * 300))
        xmax = int(np.minimum(300, box[3] * 300))

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
     
    cv2.imwrite(img_file_name, img)

    

def train(image_path, annotation_path, model_path, learning_rate = 0.001, batch_size = 1, epochs = 150, restore = False):
    print('Start training')
    glabels, gscores, glocalizations, img_feature = get_input(batch_size)
    loss = get_loss(img_feature, glabels, gscores, glocalizations, batch_size = batch_size)
    opt = get_optimizer(learning_rate, loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        saver = tf.train.Saver(max_to_keep = 5)
        step = 0
        if restore:
            print('Restoring.')
            ckpt = tf.train.latest_checkpoint(model_path)
            if ckpt:
                print('Checkpoint is valid.')
                step = int(ckpt.split('-')[1])
                saver.restore(sess, ckpt)
            if not ckpt:
                print('Checkpoint is invalid, please check it or remove --restore.')
                return
        
        for i in range(step, step + epochs):
            iter_loss = 0
            batch_nums = int(sample_nums / batch_size)
            epoch_loss = 0
            for j in range(batch_nums):
                img, bboxes, labels, raw_img, img_name = read_pic.read_pic_batch(image_path, annotation_path, batch_size, j)
                
                gtlocalizations, gtlabels, gtscores = process_ground_truth.all_layers_all_pictures_process(bboxes, labels, positive_threshold = positive_threshold)
                _, iter_loss = sess.run([opt, loss], feed_dict = {img_feature : img, glabels : gtlabels, gscores : gtscores, glocalizations : gtlocalizations})
                epoch_loss += iter_loss
                
            print('Epoch ', i, 'is finished. The loss is: ', epoch_loss / batch_nums)
            saver.save(sess, os.path.join(model_path, 'ckp'), global_step = i)
                
            #pred_result, pred_logits, pred_loc, pred_softlogits = sess.run(ssd_net.ssd_net(img))
                
            #batch_nms_locs, batch_nms_scores, batch_nms_labels = encode_predictions.batch_result_encode(pred_result, pred_logits, pred_loc, batch_size)

            #print(batch_nms_locs)
            #draw_pic.draw_box_and_save(raw_img, img_name,  batch_nms_locs, batch_nms_labels)
            
            
if __name__ == '__main__':
    with tf.Graph().as_default():
        train('/home/cxd/FDDB2VOC/JPEGImages/', '/home/cxd/FDDB2VOC/Annotations/', './save/', batch_size = 32, restore = True)
    
    
    
#                boxes = []
#                labelss = []
#                logitss = []
#                softlogitss = []
#                for i in range(6):
#                    
#                    labels = np.reshape(pred_result[i], (-1))
#                    locs = np.reshape(pred_loc[i], (-1, 4))
#                    logits = np.reshape(pred_logits[i], (-1, 2))
#                    softlogits = np.reshape(pred_softlogits[i], (-1, 2))
#                    boxes.extend(locs)
#                    labelss.extend(labels)
#                    logitss.extend(logits)
#                    softlogitss.extend(softlogits)
#                
#                boxes = np.array(boxes)
#                labelss = np.array(labelss)
#                logitss = np.array(logitss)
#                softlogitss = np.array(softlogitss)
#                    
#                boxxxxx = []
#                for i in range(8732):
#                    if labelss[i] == 1:
#                        #print(labelss[i], boxes[i], gtlocalizations[i], gtlabels[i], logitss[i], softlogitss[i])
#                        boxxxxx.append(boxes[i])
#                #draw(raw_img[0],  np.array(boxxxxx))
#                
#                #print(boxxxxx)