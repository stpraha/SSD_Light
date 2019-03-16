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

batch_size = 1

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
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        img, raw_img, img_name = read_pic.read_test_pic(batch_size, 0)

        pred_result, pred_logits, pred_loc = sess.run([predictions, logits, localizations], feed_dict = {img_feature : img})


        print(np.reshape(pred_result[0], (-1)))
        batch_nms_locs, batch_nms_scores, batch_nms_labels = encode_predictions.batch_result_encode(pred_result, pred_logits, pred_loc, batch_size)


        print(batch_nms_locs[0].shape)
        
        #print(np.reshape(batch_nms_labels, (-1)))
        draw_pic.draw_box_and_save(raw_img, img_name,  batch_nms_locs, batch_nms_labels)
        
        
if __name__ == '__main__':
    with tf.Graph().as_default():
        test()