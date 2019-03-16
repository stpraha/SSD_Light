# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:21:41 2019
@author: Stpraha
"""
import cv2
import numpy as np

voc_labels_reverse = {0: ('none', 'Background'), 3: ('aeroplane', 'Vehicle'),
              2: ('bicycle', 'Vehicle'), 1: ('bird', 'Animal'),
              4: ('boat', 'Vehicle'), 5: ('bottle', 'Indoor'),
              6: ('bus', 'Vehicle'), 7: ('car', 'Vehicle'),
              8: ('cat', 'Animal'), 9: ('chair', 'Indoor'),
              10: ('cow', 'Animal'), 11: ('diningtable', 'Indoor'),
              12: ('dog', 'Animal'), 13: ('horse', 'Animal'),
              14: ('motorbike', 'Vehicle'), 15: ('person', 'Person'),
              16: ('pottedplant', 'Indoor'), 17: ('sheep', 'Animal'),
              18: ('sofa', 'Indoor'), 19: ('train', 'Vehicle'),
              20: ('tvmonitor', 'Indoor')}

out_path = './out/'

def draw_single_pic(img, img_name, boxes, labels, pic):
    print('draw!!!!')
    img_file_name = out_path + img_name + '_ssd.jpg'
    
    for i in range(len(boxes)):
        #print(labels[i])
        label = labels[i][0]
        if label == 0:
            #print('label not able')
            continue
        label = voc_labels_reverse[label][0]

        box = boxes[i]
        ymin = int(np.maximum(0, box[0] * 300))
        xmin = int(np.maximum(0, box[1] * 300))
        ymax = int(np.minimum(300, box[2] * 300))
        xmax = int(np.minimum(300, box[3] * 300))

        print(ymin, xmin, ymax, xmax)
        
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        

        if (ymin > 10):
            img = cv2.putText(img, label, (xmin, ymin - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
        else:
            img = cv2.putText(img, label, (xmin, ymax + 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
    
        
    cv2.imwrite(img_file_name, img)


def draw_box_and_save(batch_img, img_name, batch_boxes, batch_labels):
    for i in range(len(batch_img)):
        #print('boxes shape', batch_boxes[i].shape)
        draw_single_pic(batch_img[i], img_name[i], batch_boxes[i], batch_labels[i], i)
        
        
        
        
        
        
        
        