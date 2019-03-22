# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:47:05 2018

@author: Stpraha
"""
import numpy as np
import math

"""
Input img_shape: 300 x 300
After:
    conv4  38 x 38 x 512
    conv7  19 x 19 x 1024
    conv8  10 x 10 x 512
    conv9  5 x 5 x 256
    conv10 3 x 3 x 256
    conv11 1 x 1 x 256
"""
#原始图像的大小
img_shape = (300, 300)
#从ssd_net得到的六个feature的大小
feature_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
#feature_shapes = [(37, 37), (18, 18), (9, 9), (5, 5), (3, 3), (1, 1)]
#s_k的比例，论文里是0.2最小值，0.9最大值
anchor_size_bounds = [0.10, 0.90]

anchor_ratios = [[1, 2.0/3, 0.5],
                 [1, 4.0/5, 2.0/5, 0.5, 2.0/3],
                 [1, 4.0/5, 2.0/5, 0.5, 2.0/3],
                 [1, 4.0/5, 2.0/5, 0.5, 2.0/3],
                 [1, 2.0/3, 0.5],
                 [1, 2.0/3, 0.5]]

def generate_anchor(img_shape = img_shape,
                    feature_shapes = feature_shapes,
                    anchor_ratios = anchor_ratios):
    """
        Argument:
            img_shapes: The original image shape 300*300
            feature_shapes: Shape of the feature maps
            anchor_ratios = according to the paper
        Return:
            all_anchors: all points x and y; height and width 
                         they are all scale ratio, not absolutly value
    """
    #s_max 和s_min 为比例的最大值和最小值
    s_min = anchor_size_bounds[0]
    s_max = anchor_size_bounds[1]
    #m为特征图个数，但这里设置为5，因为第一层，VGG的conv4是单独设置的。
    m = 5
    s_k = np.zeros((m + 2, ))
    #s_k = s_min + [(s_max - s_min) / (m - 1)] * (k - 1)  k取值范围: [1, m]
    step_length = math.floor((s_max - s_min) / (m - 1) * 100)

    #这里，如果layer为conv4的话，有独立的一套计算系统。
    #对于第一个计算图，其尺度比例一般设置为s_min / 2
    s_k[0]= s_min / 2
    #https://zhuanlan.zhihu.com/p/33544892
    #这种计算方式参考了SSD的caffe源码
    #因为后面要用到sqrt(s_k[i], s_k[i+1])， 所以这里s_k多计算了一个
    for i in range(m + 1):
        s_k[i+1] = s_min + step_length / 100 * i
    #得到各个特征图的先验框的尺度
    #这里考虑了一下，还是返回比例比较好，在主体里计算具体的尺度
    anchor_sizes = s_k #* img_shape[0]
    
    all_anchors = []
    #print(anchor_sizes)
    for i in range(m + 1):
        #这里，得到的anchor_bboxes的结构为：
        #   anchor_bboxes[0]: y坐标，对于300 x 300的比例
        #   anchor_bboxes[1]: x坐标，同上
        #   anchor_bboxes[2]: 框框的高度， 针对300 x 300的比例
        #   anchor_bboxes[3]: 框框的宽度，同上
        anchor_bboxes = generate_anchor_one_layer([anchor_sizes[i], anchor_sizes[i+1]],  #这里将s_k[i]  s_k[i+1]组合起来了
                                                  anchor_ratios[i],
                                                  feature_shapes[i])
        all_anchors.append(anchor_bboxes)
    #总共的anchor个数为 38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4
    #一共有8732个，与论文的数量相符合
    return all_anchors

def generate_anchor_one_layer(anchor_size,
                              anchor_ratio,
                              feature_shape,
                              offset = 0.5):
    """
        Argument:
            anchor_size: the size of the ONE anchor
            anchor_ratio: look the top
            feature_shape: the shape of the ONE feature
            offset: it makes the box center locate at the center of each unit
        Return:
            y_: the Y of the point center; not absolutly value; scale ratio
            x_: the X of ....
            height_k: the height of the BOXES of each point, 
                      one (y_, x_) corresponding to 4 or 6 heights
            width_k: the width of ...  
    """
    height_k = []
    width_k = []
    
    #以第一种为例：
    #尺寸：a_r = 1, 尺度 = s_k的正方形
    #     利用其余两个a_r 及s_k 算出尺度的长方形
    for i, ratio in enumerate(anchor_ratio):
        height_k.append(anchor_size[0] / math.sqrt(ratio))
        width_k.append(anchor_size[0] * math.sqrt(ratio))
        
    #并且还要额外加一种，那就是尺度为sqrt(s_k[i] * s_k[i+1])的正方形
    height_k.append(math.sqrt(anchor_size[0] * anchor_size[1]))
    width_k.append(math.sqrt(anchor_size[0] * anchor_size[1]))
    #width_k, height_k  tested  ok
    
    #Get x and y
    y, x = np.mgrid[0:feature_shape[0], 0:feature_shape[1]]
    #给他们加上偏置，因为每个先验框的中心是在各单元的中心的。
    y = y + offset
    x = x + offset
    #然后将其变成与feature相对应的比例。其实也就是对应于整张原始图像的位置比例
    #y, x * 300 后应该就是具体位置了。
    y_ = y / feature_shape[0]
    x_ = x / feature_shape[1]
    
    #这里将y，x增加一维，至于为什么要增加一维，参考的代码里说是为了easy broadcasting
    #暂时还没到那一步，就按照他的来
    y_ = np.expand_dims(y_, axis = -1)
    x_ = np.expand_dims(x_, axis = -1)
    height_k = np.expand_dims(height_k, axis = -1)
    width_k = np.expand_dims(width_k, axis = -1)

    return y_, x_, height_k, width_k
    

    
if __name__ == '__main__':
    result = generate_anchor()
    print(len(result))
    print(len(result[0]))
    print(result[0][0].shape)
    print(result[0][1].shape)
    





