# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:11:23 2019
@author: Stpraha
"""
import numpy as np
import tensorflow as tf
import test_ssd
import generate_anchor
np.set_printoptions(threshold = np.inf)


nms_threshold = 0.3

def flatten(pred_cls, pred_logits, pred_loc, batch_size, num_classes):
    """
    Arguments:
        pred_...: 6 x batch_size x ? x ? x ?
    Return:
        batch_...: batch ordered list. batch_size x 8732 x ?. one batch one picture
    """    
    batch_cls = []
    batch_logits = []
    batch_loc = []
    batch_scores = []
    #-----------------------------------------------------------------------------------------------------
    #Notice: logits: list of array. 6 x batch_size x may(38x38x4x21)
    #   6 is the layer amount, also the length of list.
    #   The array, membership of the list, Ex. batch_sizex38x38x4x21.
    #   Contrary to loss_function.py, here we should adjust its arrangment.
    #   logits, labels, loc should be placed according to picture.
    #------------------------------------------------------------------------------------------------------
    for i in range(batch_size):
        flatten_cls = []
        flatten_logits = []
        flatten_loc = []
        for j in range(6):    
            flatten_cls.extend(np.reshape(pred_cls[j][i], (-1)))
            flatten_logits.extend(np.reshape(pred_logits[j][i], (-1, num_classes)))
            flatten_loc.extend(np.reshape(pred_loc[j][i], (-1, 4)))
     
        flatten_cls = np.array(flatten_cls)
        flatten_logits = np.array(flatten_logits)
        flatten_loc = np.array(flatten_loc)    
        
        flatten_scores = np.max(flatten_logits, axis = 1)
        
        batch_cls.append(flatten_cls)
        batch_logits.append(flatten_logits)
        batch_loc.append(flatten_loc)
        batch_scores.append(flatten_scores)
    
    return batch_cls, batch_logits, batch_loc, batch_scores

def loc_decode(encoded_loc, batch_size):
    """
        decoded loc first. Look process_ground_truth.py for details
        Arguments:
            encoded_loc: encoded loc.
                feature_en_ycenter = (feature_ycenter - yref) / href / prior_scaling[0]
                feature_en_xcenter = (feature_xcenter - xref) / wref / prior_scaling[1]
                feature_en_h = np.log(feature_h / href) / prior_scaling[2]
                feature_en_w = np.log(feature_w / wref) / prior_scaling[3]
        Return:
            decode_loc: decoded loc. 8732 x 4, [ymin, xmin, ymax, xmax]
    """
    
    anchor_layers = generate_anchor.generate_anchor()
    
    all_layer_decoded_bboxes = []
    for i, anchor_layer in enumerate(anchor_layers):
        yref, xref, href, wref = anchor_layer
        href = href.reshape(-1)
        wref = wref.reshape(-1)
        
        batch_bboxes = []
        for j in range(batch_size):
            #get encoded y_c, x_c, h, w
            y_center = encoded_loc[i][j][:, :, :, 0]
            x_center = encoded_loc[i][j][:, :, :, 1]
            h = encoded_loc[i][j][:, :, :, 2]
            w = encoded_loc[i][j][:, :, :, 3]
            #decode to y_c, x_c, h, w
            
            prior_scaling = [0.1, 0.1, 0.2, 0.2]
            decoded_y = y_center * href * prior_scaling[0] + yref
            decoded_x = x_center * wref * prior_scaling[1] + xref
            decoded_h = np.exp(h * prior_scaling[0]) * href
            decoded_w = np.exp(w * prior_scaling[0]) * wref
            #  ---> ymin, xmin, ymax, xmax
            ymin = np.maximum(decoded_y - decoded_h / 2, 0)
            xmin = np.maximum(decoded_x - decoded_w / 2, 0)
            ymax = np.minimum(decoded_y + decoded_h / 2, 1)
            xmax = np.minimum(decoded_x + decoded_w / 2, 1)
            
            decoded_bboxes = np.stack([ymin, xmin, ymax, xmax], axis = -1)
            batch_bboxes.append(decoded_bboxes)
        batch_bboxes = np.array(batch_bboxes)
        all_layer_decoded_bboxes.append(batch_bboxes)
    
    return all_layer_decoded_bboxes


def single_label_nms(single_dets, nms_threshold):
    """
        Arguments:
            single_dets: single label dets.
            nms_threshold: threshold of nms. if IoU > nms_threshold, drop this box
        Return:
            box_keep: boxes(loc, score, label) to keep
    """
    single_dets_sort = single_dets[:,4].argsort()
    single_dets_sort = single_dets_sort[::-1]
    single_dets = single_dets[single_dets_sort]
    
    box_keep = []
    
    vols = np.maximum((single_dets[:, 2] - single_dets[:, 0]), 0) * np.maximum((single_dets[:, 3] - single_dets[:, 1]), 0)
    single_dets = single_dets[np.where(vols > 0)[0]]
        
#    for i in range(single_dets.shape[0]):
#        if(single_dets[i][5]):
#            print('ooooo', single_dets)
    
    while single_dets.shape[0] > 0:
        ymin = single_dets[:, 0]
        xmin = single_dets[:, 1]
        ymax = single_dets[:, 2]
        xmax = single_dets[:, 3]
        scores = single_dets[:, 4]
        labels = single_dets[:, 5]
        
        vols = (ymax - ymin) * (xmax - xmin)

        first_box = single_dets[0]
        box_keep.append(first_box)

        first_box_vol = vols[0]

        #get inter box
        inter_ymin = np.maximum(first_box[0], ymin[1:])
        inter_xmin = np.maximum(first_box[1], xmin[1:])
        inter_ymax = np.minimum(first_box[2], ymax[1:])
        inter_xmax = np.minimum(first_box[3], xmax[1:])
        
        #calculate inter_vol
        h = np.maximum((inter_ymax - inter_ymin), 0)
        w = np.maximum((inter_xmax - inter_xmin), 0)
        
        inter_vol = h * w
        
        #calculate IoU
        iou = inter_vol / (first_box_vol + vols[1:] - inter_vol)
 
        survived_box_index = np.where(iou < 0.3)[0]

        single_dets = single_dets[survived_box_index + 1]    
        
    box_keep = np.array(box_keep)

    return box_keep

def nms(loc, scores, labels, nms_threshold = nms_threshold):
    """
        Arguments:
            loc: 8732 x 4, ymin, xmin, ymax, xmax, one picture
            scores, 8732
            labels, 8732
        Return:
            nms_loc, nms_scores, nms_labels
    """
    #get nms_arr, 8732 x 6, from left to right, loc, scores, labels.
    #and sorted by scores, from big to small..
    dets = np.hstack([loc, 
                      np.expand_dims(scores, axis = 1), 
                      np.expand_dims(labels, axis = 1)])
    dets_sort = dets[:,5].argsort()
    dets = dets[dets_sort]
 
    split_index = []
    label = 0
    for i in range(dets.shape[0]):
        if label != dets[i, 5]:
            split_index.append(i)
            label = dets[i, 5]
            
    splited_dets = np.split(dets, split_index, axis = 0)
    
    nms_result = []
    for i, splited_det in enumerate(splited_dets):
        #if i != 0:
        box_keep = single_label_nms(splited_det, nms_threshold = nms_threshold)
        if box_keep.shape[0] != 0:
            #print(box_keep[0][5])
            nms_result.append(box_keep)
    
    nms_result = np.concatenate(nms_result)
    nms_locs, nms_scores, nms_labels = np.split(nms_result, [4, 5], axis = 1)

    return nms_locs, nms_scores, nms_labels 


def batch_result_encode(pred_cls, pred_logits, pred_loc, batch_size, num_classes):
    """
        1. decode loc. from --> encoded(y_center, x_center, h, w) --> decoded(ymin, xmin, ymax, xmax)
        2. Adjust its order. from by anchor_layers --> to by pictures.
        3. nms
        Arguments:
            All the input arguments are arrangen by anchor_layers.
            pred_cls / pred_logits: probbality of classes.
            pred_loc: predicted localizations. one batch.
        Returns:
            batch_nms_locs: a list of array. one parameter one picture.
            batch_nms_scores: similar
            batch_nms_labels: similar
    """
    #decode pred_loc --> [ymin, xmin, ymax, ]
    
    pred_loc = loc_decode(pred_loc, batch_size)
    
    batch_cls, batch_logits, batch_loc, batch_scores = flatten(pred_cls, pred_logits, pred_loc, batch_size, num_classes = num_classes)
    
    batch_nms_locs = []
    batch_nms_scores = []
    batch_nms_labels = []
    for i in range(batch_size):
        loc = batch_loc[i]
        scores = batch_scores[i]
        labels = batch_cls[i]

        nms_locs, nms_scores, nms_labels  = nms(loc, scores, labels)
        batch_nms_locs.append(nms_locs)
        batch_nms_scores.append(nms_scores)
        batch_nms_labels.append(nms_labels)
    
    return batch_nms_locs, batch_nms_scores, batch_nms_labels
    