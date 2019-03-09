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

batch_size = 2

def get_input():
    img_feature = tf.placeholder(tf.float32, [batch_size, 300, 300, 3])
    
    return img_feature


def get_prediction(img_feature):
    predictions, logits, localizations = ssd_net.ssd_net(img_feature, is_training = False)
    
    return predictions, logits, localizations

    
def test():
    print('now start testing')
    img_feature = get_input()
    predictions, logits, localizations = get_prediction(img_feature)
    
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('./save/')
    if ckpt:
        print('ckpt valid')
        
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        img, bboxes, labels, raw_img, img_name = read_pic.read_pic_batch(batch_size, 0)
        sess.run(tf.global_variables_initializer())
        pred_cls, pred_logits, pred_loc = sess.run([predictions, logits, localizations], feed_dict = {img_feature : img})

        batch_nms_locs, batch_nms_scores, batch_nms_labels = encode_predictions.batch_result_encode(pred_cls, pred_logits, pred_loc, batch_size)
        
        draw_pic.draw_box_and_save(raw_img, img_name,  batch_nms_locs, batch_nms_labels)
        
        
if __name__ == '__main__':
    with tf.Graph().as_default():
        test()
        
        
        
        
        
        
        
        
        
        
        
        