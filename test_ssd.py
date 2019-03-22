# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:18:24 2019
@author: Stpraha
"""

import tensorflow as tf
import numpy as np
import ssd_net
import read_pic
import encode_predictions
import draw_pic

class_num = 4

def get_input(batch_size):
    img_feature = tf.placeholder(tf.float32, [batch_size, 300, 300, 3])
    
    return img_feature


def get_prediction(img_feature, num_classes):
    pred_result, pred_logits, pred_loc, pred_softlogits = ssd_net.ssd_net(img_feature,  num_classes, is_training = False)
    
    return pred_result, pred_logits, pred_loc, pred_softlogits

    
def test(image_path, out_path, model_path, batch_size = 1):
    print('Start testing')
    img_feature = get_input(batch_size)
    pred_result, pred_logits, pred_loc, pred_softlogits = get_prediction(img_feature, num_classes = class_num)
    
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(model_path)
    if ckpt:
        print('Checkpoint is valid.')
    if not ckpt:
        print('Checkpoint is invalid, please check it.')
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        img, raw_img, img_name = read_pic.read_test_pic(image_path, batch_size, 0)
 
        result, logits, loc, softlogits = sess.run([pred_result, pred_logits, pred_loc, pred_softlogits], feed_dict = {img_feature : img})
    
        batch_nms_locs, batch_nms_scores, batch_nms_labels = encode_predictions.batch_result_encode(result, logits, loc, batch_size, num_classes = class_num)
        draw_pic.draw_box_and_save(raw_img, img_name,  batch_nms_locs, batch_nms_labels, out_path)
        
        
if __name__ == '__main__':
    with tf.Graph().as_default():
        test('/home/cxd/SSD_Light/demo/', '/home/cxd/SSD_Light/out/', './save/', batch_size = 32)