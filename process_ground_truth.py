import generate_anchor
import read_pic
import numpy as np
import encode_predictions
import cv2
np.set_printoptions(threshold=np.inf) 


def calculate_jaccard(ymin, xmin, ymax, xmax, single_bbox):
    """
        Jaccard score: J(A, B) = (A inter B) / (A union B)
        Arguments:
            yim, xmin, ymax, xmax: upper/left/down/right boundary of anchor.
                                    It's shape is accroding to the layer.
            single_bbox: single ground truth bbox. 1 x 4
        Return:
            jaccard: the jaccard scores between all anchors (one layer) and single_bbox.
                     It's shape is the same as ymin/xmin...
    """  
    #calculate Intersection
    #single_bbox: 1 x 4    ymin: according to layer. ex. 3x3x4 
    inter_ymin = np.maximum(ymin, single_bbox[0])
    inter_xmin = np.maximum(xmin, single_bbox[1])
    inter_ymax = np.minimum(ymax, single_bbox[2])
    inter_xmax = np.minimum(xmax, single_bbox[3])

    h = np.maximum((inter_ymax - inter_ymin), 0)
    w = np.maximum((inter_xmax - inter_xmin), 0)

    inter_vol = h * w
    
    #calculate Union
    #A union B = A + B - (A inter B)
    bbox_vol = (single_bbox[2] - single_bbox[0]) * (single_bbox[3] - single_bbox[1])
    anchor_vol = (xmax - xmin) * (ymax - ymin)
    union_vol = bbox_vol + anchor_vol - inter_vol
    
    #calculate jaccard
    jaccard = np.divide(inter_vol, union_vol)
    
    return jaccard


def one_layer_one_picture_bboxes_process(anchor_layer, bbox, label, layer_index, positive_threshold):
    """
        Each layer(one of 6 anchor_layer), one picture's bboxes.
        Arguments:
            anchor_layer: single layer (part of anchor_layers)
            bbox: single pic bboxes. bbox_per_pic x 4
            lael: single pic labels. bbox_per_pic x 1
        Return:
            single_pic_localizations: array, (according to layer) x 4
            single_pic_labels: array, (according to layer)
            single_pic_scores: array, (according to layer)
    """
    yref, xref, href, wref = anchor_layer
    href = href.reshape(-1)
    wref = wref.reshape(-1)
    
    ymin = yref - href / 2.  #upper boundary
    xmin = xref - wref / 2.  #left boundary
    ymax = yref + href / 2.  #
    xmax = xref + wref / 2.  #
        
    shape = ymin.shape
    
    feature_labels = np.zeros(shape)
    feature_scores = np.zeros(shape)
    feature_ymin = np.zeros(shape)
    feature_xmin = np.zeros(shape)
    feature_ymax = np.ones(shape)
    feature_xmax = np.ones(shape)

    #------------------------------------------------------------------------------------
    #This loop does have some thing special. Take feature_labels as example.
    #At the begining, feature_scores == 0. After loop, feature_scores is 
    #almost impossible to be zero. The jaccard score continue updates. Using mask,
    #the value of feature_labels --> the label(class) with higher jaccard score. 
    #It will not change if new jaccard score is less than existing feature_score.
    #As the loop goes on. The feature_labels --> label(class) with highest jaccard score.
    #Same as feature_labels, other 4 'array' sotre the highest jaccard score
    #label(class)'s localization.
    #------------------------------------------------------------------------------------
    for i, single_bbox in enumerate(bbox):
        single_label = label[i]
        jaccard = calculate_jaccard(ymin, xmin, ymax, xmax, single_bbox)
        
        mask = np.logical_and(jaccard > positive_threshold, np.greater(jaccard, feature_scores))
        #mask =  np.greater(jaccard, feature_scores)
        imask = mask.astype(int)
        fmask = mask.astype(float)
        
        feature_labels = imask * single_label + (1 - imask) * feature_labels
        
        feature_scores = np.where(mask, jaccard, feature_scores) #actually, same as jaccard..
        
        feature_ymin = fmask * single_bbox[0] + (1 - fmask) * feature_ymin
        feature_xmin = fmask * single_bbox[1] + (1 - fmask) * feature_xmin
        feature_ymax = fmask * single_bbox[2] + (1 - fmask) * feature_ymax
        feature_xmax = fmask * single_bbox[3] + (1 - fmask) * feature_xmax
    
    feature_ycenter = (feature_ymax + feature_ymin) / 2
    feature_xcenter = (feature_xmax + feature_xmin) / 2
    feature_h = (feature_ymax - feature_ymin)
    feature_w = (feature_xmax - feature_xmin)
    
    #---------------------------
    #
    #CAUTION!
    #---------------------------

    prior_scaling = [0.1, 0.1, 0.2, 0.2]
    feature_en_ycenter = (feature_ycenter - yref) / href / prior_scaling[0]
    feature_en_xcenter = (feature_xcenter - xref) / wref / prior_scaling[1]
    feature_en_h = np.log(feature_h / href) / prior_scaling[2]
    feature_en_w = np.log(feature_w / wref) / prior_scaling[3]
    
    feature_localizations = np.stack([feature_en_ycenter, 
                                      feature_en_xcenter, 
                                      feature_en_h, 
                                      feature_en_w])
    
    feature_localizations = np.transpose(feature_localizations, (1, 2, 3, 0))
    
    return feature_localizations, feature_labels, feature_scores


def one_layer_all_pictures_process(anchor_layer, bboxes, labels, layer_index, positive_threshold):
    """
        Each layer(one of 6 anchor_layer), one batch pictures.
        Arguments:
            anchor_layer: single layer (part of anchor_layers)
            bboxes: all(one batch) ground truth bboxes. batch_size x bbox_per_pic x 4
            labels: all(one batch) ground truth labels. batch_size x label_per_pic
        Return:
            single_layer_localizations: list, batch_size x (according to layer) x 4
            single_layer_labels: list, batch_size x (according to layer)
            single_layer_scores: list, batch_size x (according to layer)
    """
    single_layer_localizations = []
    single_layer_labels = []
    single_layer_scores = []
    
    for i, bbox in enumerate(bboxes):
        single_picture_localizations, single_picture_labels, single_picture_scores = one_layer_one_picture_bboxes_process(anchor_layer, bboxes[i], labels[i], layer_index, positive_threshold)

        single_layer_localizations.append(single_picture_localizations)
        single_layer_labels.append(single_picture_labels)
        single_layer_scores.append(single_picture_scores)

    return single_layer_localizations, single_layer_labels, single_layer_scores

      
def all_layers_all_pictures_process(bboxes, labels, positive_threshold):
    """
        All anchor_layers, one batch pictures.
        Arguments:
            bboxes: all ground truth bboxes. batch_size x bbox_per_pic x 4     (ymin, xmin, ymax, xmax)
            labels: all ground truth labels. batch_size x bbox_pre_pic
            anchor_layers: 6 layers created by generate_anchor.
        Return:
            glocalizations: array, (batch_size x 8732) x 4
            glabels: array, (batch_size x 8732) x 1
            gscores: array, (batch_size x 8732) x 1
    """
    anchor_layers = generate_anchor.generate_anchor()
    all_localizations = []
    all_labels = []
    all_scores = []
    
    #-------------------------------------------------------------------------------------
    #Notice: glocalizations, glabels, gscores are ordered by LAYER, not picture.
    #   The first dimension is layer. The second dimension is picture of batch.
    #   Attention need to be paid when calculate loss.
    #-------------------------------------------------------------------------------------
    for i, anchor_layer in enumerate(anchor_layers):    
        single_layer_localizations, single_layer_labels, single_layer_scores = one_layer_all_pictures_process(anchor_layer, bboxes, labels, layer_index = i, positive_threshold = positive_threshold)
        
#        if i == 5:
#            r = single_layer_labels[0]
#            r = r.sum(axis = 2)
#            print(r)
        all_localizations.append(single_layer_localizations)
        all_labels.append(single_layer_labels)
        all_scores.append(single_layer_scores)
    
    glocalizations = []
    glabels = []
    gscores = []

    #flatten
    for i, localization in enumerate(all_localizations):
        flatten_localization = np.reshape(all_localizations[i], (-1, 4))
        flatten_labels = np.reshape(all_labels[i], (-1, 1))
        flatten_scores = np.reshape(all_scores[i], (-1, 1))
        
        glocalizations.extend(flatten_localization)
        glabels.extend(flatten_labels)
        gscores.extend(flatten_scores)
        
    glocalizations = np.array(glocalizations, dtype = np.float32)
    glabels = np.array(glabels, dtype = np.int32)
    gscores = np.array(gscores, dtype = np.float32)

    #print(glabels.shape)
    return glocalizations, glabels, gscores

def test():
    img, bboxes, labels, image_set, img_name_set = read_pic.read_pic_batch('F:\\VOC2007\\JPEGImages\\', 'F:\\VOC2007\\Annotations\\', 1, 0)
    #print(bboxes)
    glocalizations, glabels, gscores = all_layers_all_pictures_process(bboxes, labels)

if __name__ == '__main__':
    test()