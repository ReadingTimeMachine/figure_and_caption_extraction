from glob import glob

# do we need both of these?...
import xml.etree.ElementTree as ET
from lxml import etree

from scipy import stats
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
import pickle
import os
from PIL import Image
import cv2 as cv
import regex
import re
import pytesseract

# stuff
import config
from general_utils import parse_annotation, isRectangleOverlap, manhattan, iou_orig


depth_vec = config.depth_vec
versions = config.versions
width_vec = config.width_vec
image_size = config.IMAGE_H # assume square I guess?
threshold = config.threshold
max_boxes = config.max_boxes

def parse_annotations_to_labels(classDir_main_to, testListFile, 
                                benchmark=False, return_max_boxes=False):
    # from test file instead (from Google Collab splits)
    if not benchmark:
        with open(testListFile, 'r') as f:
            fl = f.readlines()
    else: # grab full list from directory
        fl = glob(classDir_main_to+'*xml')
    
    annotations = []
    for f in fl:
        if len(glob(classDir_main_to + f.split('/')[-1].strip())) > 0: # check for file exists --> bug checking
            annotations.append(classDir_main_to + f.split('/')[-1].strip())
    annotations2 = glob(classDir_main_to + '*')
    # sort
    annotations = np.unique(annotations).tolist()
    annotations.sort()
    
    #print(annotations)

    # NEXT: do a quick test run-through of the data generator for train/test splits
    X_full = np.array(annotations)
    Y_full_str = np.array([]) # have to loop and give best guesses for the pages that have multiple images/classes in them
    slabels = []; maxboxes = -1
    for X in X_full:
        tree = ET.parse(X)
        tags = []
        for elem in tree.iter(): 
            if 'object' in elem.tag or 'part' in elem.tag:                  
                for attr in list(elem):
                    if 'name' in attr.tag:
                        if attr.text is not None:
                            tags.append(attr.text)
                            slabels.append(attr.text)
        if len(tags)>0: 
            modeClass = stats.mode(tags).mode[0] # most frequent class that pops up on this page
            Y_full_str = np.append(Y_full_str, modeClass) # class in string
        maxboxes = max([len(tags),maxboxes])

    # NOTE: you need the full range of annotions to get ALL the labels:
    Y_full_str2 = np.array([]) # have to loop and give best guesses for the pages that have multiple images/classes in them
    slabels2 = []
    for X in annotations2:
        tree = ET.parse(X)
        tags = []
        for elem in tree.iter(): 
            if 'object' in elem.tag or 'part' in elem.tag:                  
                for attr in list(elem):
                    if 'name' in attr.tag:
                        if attr.text is not None:
                            tags.append(attr.text)
                            slabels2.append(attr.text)
        if len(tags) > 0:
            modeClass = stats.mode(tags).mode[0] # most frequent class that pops up on this page
            Y_full_str2 = np.append(Y_full_str2, modeClass) # class in string

    LABELS = np.unique(slabels2).tolist()
    CLASS = len(LABELS)
    #if use_only_one_class: CLASS=1

    # strings to integers
    Y_full = []
    labels = np.arange(len(LABELS))

    for i in range(len(Y_full_str)):
        Y_full.append( labels[np.array(LABELS) == Y_full_str[i]][0] +1 ) # 0 means unlabeled data
        if len(labels[np.array(LABELS) == Y_full_str[i]]) > 1:
            print('We have an issue!!')
            import sys
            sys.exit()

    Y_full = np.array(Y_full)
    
    if not return_max_boxes:
        return LABELS, labels, slabels, CLASS, annotations, Y_full
    else:
        return LABELS, labels, slabels, CLASS, annotations, Y_full, maxboxes
        


# ------- not 100% sure if we uses these -----------
num_cluster = 9 # has to be 9

def iou(boxes, clusters):  # 1 box -> k clusters
    n = boxes.shape[0] # number of boxes
    k = num_cluster

    box_area = boxes[:, 0] * boxes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    return inter_area / (box_area + cluster_area - inter_area)

def avg_iou(boxes, clusters):
    accuracy = np.mean([np.max(iou(boxes, clusters), axis=1)])
    return accuracy

def generator(boxes, k, dist=np.median):
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))
    clusters = boxes[np.random.choice(box_number, k, replace=False)]  # init k clusters
    while True:
        distances = 1 - iou(boxes, clusters)

        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
        last_nearest = current_nearest

    return clusters


########. def used ###############

## get number of features from a file
def get_n_features(classDir_main_to_imgs):
    # open random data object, get number of features
    t = glob(classDir_main_to_imgs + '*')[0]

    with open(t, 'rb') as f:
        arr = np.load(f)['arr_0']

    n_features = arr.shape[-1]
    return n_features

initializer = tf.random_normal_initializer(stddev=0.01)
l2 = tf.keras.regularizers.l2(4e-5)


def conv(x, filters, k=1, s=1):
    if s == 2:
        x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    else:
        padding = 'same'
    x = layers.Conv2D(filters, k, s, padding, use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization(momentum=0.03)(x)
    x = layers.Activation(tf.nn.swish)(x)
    return x


def residual(x, filters, add=True):
    inputs = x
    if add:
        x = conv(x, filters, 1)
        x = conv(x, filters, 3)
        x = inputs + x
    else:
        x = conv(x, filters, 1)
        x = conv(x, filters, 3)
    return x


def csp(x, filters, n, add=True):
    y = conv(x, filters // 2)
    for _ in range(n):
        y = residual(y, filters // 2, add)

    x = conv(x, filters // 2)
    x = layers.concatenate([x, y])

    x = conv(x, filters)
    return x


class Predict(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, anchors, CLASS, **kwargs):
        #tf.compat.v1.enable_eager_execution()
        y_pred = [(inputs[0], anchors[6:9]),
                  (inputs[1], anchors[3:6]),
                  (inputs[2], anchors[0:3])]
        #print('------')
        #print(y_pred)
        
#         print('**********')
#         y_pred2 = []
#         for y in y_pred:
#             print(y[0],y[1])
#             y_pred2.append(y[1])
#         print('*********')

        boxes_list, conf_list, prob_list = [], [], []
        #for result in [process_layer(feature_map, anchors) for (feature_map, anchors) in y_pred]:
        for result in [process_layer(feature_map, anchors,CLASS) for (feature_map, anchors) in y_pred]:
            #print('result', result)
            x_y_offset, box, conf, prob = result
            grid_size = tf.shape(x_y_offset)[:2]
            box = tf.reshape(box, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf = tf.reshape(conf, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob = tf.reshape(prob, [-1, grid_size[0] * grid_size[1] * 3, CLASS])
            boxes_list.append(box)
            conf_list.append(tf.sigmoid(conf))
            prob_list.append(tf.sigmoid(prob))

        boxes = tf.concat(boxes_list, axis=1)
        conf = tf.concat(conf_list, axis=1)
        prob = tf.concat(prob_list, axis=1)

        center_x, center_y, w, h = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - w / 2
        y_min = center_y - h / 2
        x_max = center_x + w / 2
        y_max = center_y + h / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
        #print('-----')
        #print(boxes, conf * prob)
        #print('------')

        outputs = tf.map_fn(fn=compute_nms,
                            elems=[boxes, conf * prob],
                            dtype=['float32', 'float32', 'int32'],
                            parallel_iterations=100)
        #data = fnames.map(process_path)
        #to
        #data = fnames.map(lambda x: tf.py_function(process_path, [x], [tf.string]))

        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], max_boxes, 4),
                (input_shape[1][0], max_boxes),
                (input_shape[1][0], max_boxes), ]

    def compute_mask(self, inputs, mask=None):
        return (len(inputs) + 1) * [None]
    
    
#### orig build model ####
def build_model(n_features, anchors, version, CLASS, training=True, use_ps=False): 
    #print(n_features,anchors,version,CLASS)
    depth = depth_vec[versions.index(version)]
    width = width_vec[versions.index(version)]

    # JPN add pseudo coloring
    if use_ps:
        inputs = tf.keras.layers.Input((config.IMAGE_H, config.IMAGE_W, n_features), dtype='float32')
        bn = tf.keras.layers.BatchNormalization(input_shape=[config.IMAGE_H, config.IMAGE_W, n_features], name='pseudoIn')(inputs)
        # add pseudo coloring
        pseudo_conv1 = keras.layers.Conv2D(filters=50, kernel_size=(1,1), strides=1, padding="SAME", activation="relu", 
                                input_shape=[None, None, 1], name='pseudo1')(bn) # into 20 psedo color channels
        pseudo_conv2 = keras.layers.Conv2D(filters=3, kernel_size=(1,1), strides=1, padding="SAME", activation="relu", 
                               name='pseudo2')(pseudo_conv1) # down to RGB channels
        # now back to our regularly scheduled program
        x = tf.nn.space_to_depth(pseudo_conv2, 2)
    else:
        #inputs = layers.Input([config.IMAGE_H, config.IMAGE_W, 3])
        inputs = layers.Input([config.IMAGE_H, config.IMAGE_W, n_features])
        x = tf.nn.space_to_depth(inputs, 2)    
    
    #inputs = layers.Input([config.IMAGE_H, config.IMAGE_W, n_features])
    #x = tf.nn.space_to_depth(inputs, 2)
    x = conv(x, int(round(width * 64)), 3)
    x = conv(x, int(round(width * 128)), 3, 2)
    x = csp(x, int(round(width * 128)), int(round(depth * 3)))

    x = conv(x, int(round(width * 256)), 3, 2)
    x = csp(x, int(round(width * 256)), int(round(depth * 9)))
    x1 = x

    x = conv(x, int(round(width * 512)), 3, 2)
    x = csp(x, int(round(width * 512)), int(round(depth * 9)))
    x2 = x

    x = conv(x, int(round(width * 1024)), 3, 2)
    x = conv(x, int(round(width * 512)), 1, 1)
    x = layers.concatenate([x,
                            tf.nn.max_pool(x, 5,  1, 'SAME'),
                            tf.nn.max_pool(x, 9,  1, 'SAME'),
                            tf.nn.max_pool(x, 13, 1, 'SAME')])
    x = conv(x, int(round(width * 1024)), 1, 1)
    x = csp(x, int(round(width * 1024)), int(round(depth * 3)), False)

    x = conv(x, int(round(width * 512)), 1)
    x3 = x
    x = layers.UpSampling2D()(x)
    x = layers.concatenate([x, x2])
    x = csp(x, int(round(width * 512)), int(round(depth * 3)), False)

    x = conv(x, int(round(width * 256)), 1)
    x4 = x
    x = layers.UpSampling2D()(x)
    x = layers.concatenate([x, x1])
    x = csp(x, int(round(width * 256)), int(round(depth * 3)), False)
    #p3 = layers.Conv2D(3 * (len(config.class_dict) + 5), 1, name=f'p3_{len(config.class_dict)}',
    #                   kernel_initializer=initializer, kernel_regularizer=l2)(x)
    p3 = layers.Conv2D(3 * (CLASS + 5), 1, name=f'p3_{CLASS}',
                       kernel_initializer=initializer, kernel_regularizer=l2)(x)

    x = conv(x, int(round(width * 256)), 3, 2)
    x = layers.concatenate([x, x4])
    x = csp(x, int(round(width * 512)), int(round(depth * 3)), False)
    p4 = layers.Conv2D(3 * (CLASS + 5), 1, name=f'p4_{CLASS}',
                       kernel_initializer=initializer, kernel_regularizer=l2)(x)

    x = conv(x, int(round(width * 512)), 3, 2)
    x = layers.concatenate([x, x3])
    x = csp(x, int(round(width * 1024)), int(round(depth * 3)), False)
    p5 = layers.Conv2D(3 * (CLASS + 5), 1, name=f'p5_{CLASS}',
                       kernel_initializer=initializer, kernel_regularizer=l2)(x)

    if training:
        return tf.keras.Model(inputs, [p5, p4, p3])
    else:
        return tf.keras.Model(inputs, Predict()([p5, p4, p3],anchors,CLASS))
    
    
def process_box(boxes, labels,anchors,CLASS):
    '''
    labels: integer numbers associated with labels, JPN: does this start at 0 or 1??
    boxes: [number of boxes on page, xmin, ymin, xmax, ymax]
    returns: the correctly formatted 3 y-trues
    '''
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #anchors = anchors
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    box_size = boxes[:, 2:4] - boxes[:, 0:2]

    y_true_1 = np.zeros((image_size // 32,
                            image_size // 32,
                            3, 5 + CLASS), np.float32)
    y_true_2 = np.zeros((image_size // 16,
                            image_size // 16,
                            3, 5 + CLASS), np.float32)
    y_true_3 = np.zeros((image_size // 8,
                            image_size // 8,
                            3, 5 + CLASS), np.float32)

    y_true = [y_true_1, y_true_2, y_true_3]

    box_size = np.expand_dims(box_size, 1)

    min_np = np.maximum(- box_size / 2, - anchors / 2)
    max_np = np.minimum(box_size / 2, anchors / 2)

    whs = max_np - min_np

    overlap = whs[:, :, 0] * whs[:, :, 1]
    union = box_size[:, :, 0] * box_size[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10

    iou = overlap / union
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        feature_map_group = 2 - idx // 3
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_size[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        try:
            #y_true[feature_map_group][y, x, k, 5 + c] = 1.
            y_true[feature_map_group][y, x, k, 5 + c -1] = 1. # labels start at 0
        except:
            print('in parse')
            print(y,x,k,c, 5+c)
            print(labels)

    return y_true_1, y_true_2, y_true_3

def process_layer(feature_map, anchors,CLASS):
    #print('process layer', feature_map)
    grid_size = tf.shape(feature_map)[1:3]
    ratio = tf.cast(tf.constant([config.IMAGE_H, config.IMAGE_W]) / grid_size, tf.float32)
    rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

    feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + CLASS])
    #print('ratio', ratio)

    box_centers, box_sizes, conf, prob = tf.split(feature_map, [2, 2, 1, CLASS], axis=-1)
    box_centers = tf.nn.sigmoid(box_centers)

    grid_x = tf.range(grid_size[1], dtype=tf.int32)
    grid_y = tf.range(grid_size[0], dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * ratio[::-1]

    box_sizes = tf.exp(box_sizes) * rescaled_anchors
    box_sizes = box_sizes * ratio[::-1]

    boxes = tf.concat([box_centers, box_sizes], axis=-1)
    #print('in process layer')
    #print(x_y_offset, boxes, conf, prob)

    return x_y_offset, boxes, conf, prob


def box_iou(pred_boxes, valid_true_boxes):
    #batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)
    pred_box_xy = tf.cast(pred_boxes[..., 0:2], tf.float32)
    pred_box_wh = tf.cast(pred_boxes[..., 2:4], tf.float32)

    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    true_box_xy = tf.cast(valid_true_boxes[:, 0:2], tf.float32)
    true_box_wh = tf.cast(valid_true_boxes[:, 2:4], tf.float32)
    
    intersect_min = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
    intersect_max = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)

    intersect_wh = tf.maximum(intersect_max - intersect_min, 0.)

    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    return intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)


def compute_nms(args):
    boxes, classification = args

    def nms_fn(score, label):
        score_indices = tf.where(tf.keras.backend.greater(score, threshold))

        filtered_boxes = tf.gather_nd(boxes, score_indices)
        filtered_scores = tf.keras.backend.gather(score, score_indices)[:, 0]

        nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_boxes, 0.1)
        score_indices = tf.keras.backend.gather(score_indices, nms_indices)

        label = tf.gather_nd(label, score_indices)
        score_indices = tf.keras.backend.stack([score_indices[:, 0], label], axis=1)

        return score_indices

    all_indices = []
    for c in range(int(classification.shape[1])):
        scores = classification[:, c]
        labels = c * tf.ones((tf.keras.backend.shape(scores)[0],), dtype='int64')
        all_indices.append(nms_fn(scores, labels))
    indices = tf.keras.backend.concatenate(all_indices, axis=0)

    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=tf.keras.backend.minimum(max_boxes, tf.keras.backend.shape(scores)[0]))

    indices = tf.keras.backend.gather(indices[:, 0], top_indices)
    boxes = tf.keras.backend.gather(boxes, indices)
    labels = tf.keras.backend.gather(labels, top_indices)

    pad_size = tf.keras.backend.maximum(0, max_boxes - tf.keras.backend.shape(scores)[0])

    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = tf.keras.backend.cast(labels, 'int32')

    boxes.set_shape([max_boxes, 4])
    scores.set_shape([max_boxes])
    labels.set_shape([max_boxes])

    return [boxes, scores, labels]
#################################################################
# def build_predict(weightsFile, anchorsFile, classDir_main_to_imgs, 
#                   LABELS,version='l', debug=False, use_ps = False):
#     # read in anchors
#     with open(anchorsFile, 'rb') as f:
#         anchors = pickle.load(f) 
#         anchors = anchors.astype('float32')
#     # get number of features
#     n_features = get_n_features(classDir_main_to_imgs)
#     if debug:
#         print('n features=', n_features)
    
#     model_predict = build_model(n_features, anchors, version, len(LABELS),training=False, use_ps=False)
#     if debug:
#         tf.keras.utils.plot_model(model_predict, "yolo_v5.png", show_shapes=True, 
#                                   show_layer_names=True, expand_nested=False)
#     model_predict.load_weights(weightsFile) # note there was a True) here and i took it out and now things work?  for REASONS.
    
#     return model_predict



############# LOAD ANNOTATIONS #####################
def get_true_boxes(a,LABELS, badskews, badannotations, 
                   annotation_dir='', feature_dir='',
                  check_for_file=True):
    # get annotations, use pdffigures2 boxes as well
    years_ind = []
    imgs_name, true_boxes, pdfboxes, \
       pdfrawboxes = parse_annotation([a], LABELS,feature_dir = feature_dir,
                     annotation_dir = annotation_dir, parse_pdf=True,
                                     check_for_file=check_for_file)    

    if len(true_boxes) > 0:
        truebox = true_boxes[0] # formatting mess
    else:
        truebox = []
    # no idea, but some formatting things I need to fix
    if len(pdfboxes)> 0:
        pdfboxes = pdfboxes[0]
    else:
        pdfboxes = []
    if len(pdfrawboxes)> 0:
        pdfrawboxes = pdfrawboxes[0]
    else:
        pdfrawboxes = []
        
    # get tables + captions
    tables = []; tcaptions = []; pdfbsave = []
    #import sys; sys.exit()
    for b in pdfboxes:
        #print(b)
        if b[-1] == 'table':
            tables.append(b)
        elif b[-1] == 'table caption':
            tcaptions.append(b)
        else:
            pdfbsave.append(b)
    # check distance
    if len(tcaptions) > 0:
        # also check for pairs
        if len(tcaptions) == len(tables):
            for iit,t in enumerate(tables):
                mdist = 1e5
                tcb = -1
                for itc,tc in enumerate(tcaptions):
                    md = manhattan(t[:-1],tc[:-1])
                    if md < mdist:
                        mdist = md
                        tcb = tc
                # combine boxes
                tables[iit] = [ min([t[0],tcb[0]]), min([t[1],tcb[1]]), 
                               max([t[2],tcb[2]]), max([t[3],tcb[3]]) , t[-1]]
    for t in tables:
        pdfbsave.append(t)
    pdfboxes = pdfbsave
        

    # look for image
    #print(imgs_name)
    iiname = imgs_name[0]
    iiname = iiname[:iiname.rfind('.')]
    #print(iiname)
    if (iiname.split('/')[-1] in badskews) or (iiname.split('/')[-1] in badannotations):
        print('bad skew or annotation for', icombo)
        ##continue
    if check_for_file:
        if os.path.isfile(iiname+'.jpeg'):
            imgName = iiname + '.jpeg'
        elif os.path.isfile(iiname+'.jpg'):
            imgName = iiname + '.jpg'
        else:       
            imgName = glob(iiname+'*')[0]
    pdfbase = a.split('/')[-1].split('_p')[0]

    if len(pdfbase) > 0 and pdfbase[:4].isdigit():
        # store years
        year = int(pdfbase[:4])
    else:
        year = -1

    years_ind.append(year)
    
    return imgs_name, pdfboxes, pdfrawboxes,years_ind, truebox


########### LOAD IMAGE FEATURES AND IMAGE DATA #####################
def get_ocr_results(imgs_name, dfMakeSense,dfsave,
                    use_tfrecords=True, image_np=None,
                   width=None,height=None,images_jpeg_dir=None):
    if images_jpeg_dir is None: images_jpeg_dir=config.images_jpeg_dir
    
    goOn = True
    if dfMakeSense is not None:
        # feature file -- if not given
        if not use_tfrecords:
            image_np = np.load(imgs_name[0])['arr_0']
            image_np = image_np.astype(np.float32) / 255.0 


        # OCR file/data
        ff = imgs_name[0].split('/')[-1].split('.npz')[0]
        # get height, width of orig image
        try:
            indff = dfMakeSense.loc[dfMakeSense['filename']==ff].index[0]
        except:
            print('we have an issue here!!')
            print(ff)
            import sys; sys.exit()
        # reshape to this
        dfMS = dfMakeSense.loc[dfMakeSense['filename']==ff]
        #fracx = msw[indff]*1.0/image_np.shape[1]
        #fracy = msh[indff]*1.0/image_np.shape[0]
        fracx = dfMS['w']*1.0/image_np.shape[1]
        fracy = dfMS['h']*1.0/image_np.shape[0]
        # check for file
        if os.path.isfile(images_jpeg_dir+ff+'.jpeg'):
            indh = ff+'.jpeg'
        elif os.path.isfile(images_jpeg_dir+ff+'.jpg'):
            indh = ff+'.jpg'
        else:
            # find correct hocr index
            ff = glob(images_jpeg_dir+ff + '*')
            if len(ff) == 0:
                print('have issue! here in looking for thing.')
                import sys; sys.exit()
            else:
                indh = ff[0].split('/')[-1]                
    elif (width is not None) and (height is not None):
        fracx = width/image_np.shape[1]
        fracy = height/image_np.shape[0]
        indh = dfsave.index.values[0]
    # grab OCR
    #print(dfsave)
    try:
        hocr = dfsave.loc[indh]['hocr']
    except:
        print('NO hocr for', indh)
        goOn = False
        
    if goOn:
        # paragraphs from OCR
        bbox_par = []
        nameSpace = ''
        for l in hocr.split('\n'):
            if 'xmlns' in l:
                nameSpace = l.split('xmlns="')[1].split('"')[0]
                break
        ns = {'tei': nameSpace}
        tree = etree.fromstring(hocr.encode())
        # get paragraphs
        lines = tree.xpath("//tei:p[@class='ocr_par']/@title", namespaces=ns)
        langs = tree.xpath("//tei:p[@class='ocr_par']/@lang", namespaces=ns)
        for l,la in zip(lines,langs):
            x = l.split(' ')
            b = np.array(x[1:]).astype('int')
            area = (b[3]-b[1])*(b[2]-b[0])
            bbox_par.append((b,area,la))

        # get words
        bboxes_words = []
        lines = tree.xpath("//tei:span[@class='ocrx_word']/@title", namespaces=ns)
        text = tree.xpath("//tei:span[@class='ocrx_word']/text()", namespaces=ns)
        # get words
        for t,l in zip(text,lines):
            x = l.split(';') # each entry
            for y in x:
                if 'bbox' in y:
                    z = y.strip()
                    arr=y.split()
                    b = np.array(arr[1:]).astype('int')
                elif 'x_wconf' in y:# also confidence
                    c = y.split('x_wconf')[-1].strip()
            bboxes_words.append((b,t,int(c))) 

        # squares from image processing
        # grab squares
        shere = dfsave.loc[indh]['squares']    
        bbsq = []
        for ss in shere:
            if len(ss) > 0:
                x,y,w,h = cv.boundingRect(ss) # in orig page
                bbsq.append((x,y,x+w,y+h))
                
        cbhere = dfsave.loc[indh]['colorbars']    
        cbsq = []
        for ss in cbhere:
            if len(ss) > 0:
                x,y,w,h = cv.boundingRect(ss) # in orig page
                cbsq.append((x,y,x+w,y+h))

        # grab rotation
        rotation = dfsave.loc[indh]['rotation']
        
        #---------------- image processing figure captions -----------------------
        # some of this is repeat, but for finding captions with image processing:
        bbox_hocr = []
        for i,word in enumerate(tree.xpath("//tei:span[@class='ocrx_word']", namespaces=ns)):
            myangle = word.xpath("../@title", namespaces=ns) # this should be line tag
            par = word.xpath("./@title", namespaces=ns)[0]
            bb = np.array(par.split(';')[0].split(' ')[1:]).astype('int').tolist()
            c = int(par.split('x_wconf')[-1])
            t = word.xpath("./text()",namespaces=ns)[0]
            if len(myangle) > 1:
                print('HAVE TOO MANY PARENTS')
            if 'textangle' in myangle[0]:
                # grab text angle
                textangle = float(myangle[0].split('textangle')[1].split(';')[0])
            else:
                textangle = 0.0
            # find associated paragraph
            par = word.xpath("../../@title", namespaces=ns)[0]
            bbpar = np.array(par.split(';')[0].split(' ')[1:]).astype('int').tolist()
            # and carea
            car = word.xpath("../../../@title", namespaces=ns)[0]
            bbcar = np.array(car.split(';')[0].split(' ')[1:]).astype('int').tolist()
            # replace a few obvious things
            # this cleaning should be layed out in a config
            text = t.replace('Fic.', 'Fig.')
            text = text.replace('Frc.', 'Fig.')
            text = text.replace('FIC.', 'FIG.')
            if len(text) == 3:
                text = text.replace('Fic', 'Fig.')

            # put in x,y,w,h format
            bb[2] = bb[2]-bb[0]; bb[3] = bb[3]-bb[1]
            bbpar[2] = bbpar[2]-bbpar[0]; bbpar[3] = bbpar[3]-bbpar[1]
            bbcar[2] = bbcar[2]-bbcar[0]; bbcar[3] = bbcar[3]-bbcar[1]
            bbox_hocr.append((bb,t,c,textangle,bbpar,bbcar))  
            
        # error checking: find rotation two ways -- this should also be in the annotation generation step
        rots = []
        for (x,y,w,h),text,c,r,_,_ in bbox_hocr:
            if c>=config.ccut_rot:
                if (len(text) > 0):  
                    rots.append(r)
        
        if len(rots) > 0:
            rotatedAngleOCR = stats.mode(rots).mode[0]
        else: # if no words, assume not rotated
            rotatedAngleOCR = 0
             
        backtorgb = np.array(Image.open(images_jpeg_dir+indh).convert('RGB'))
        #print(config.images_jpeg_dir+indh) 
        try:
            newdata=pytesseract.image_to_osd(backtorgb.copy())
            rotationImage = int(re.search('(?<=Rotate: )\d+', newdata).group(0))
        except:
            print('something bad has happened with image_to_osd on', indh)
            rotationImage = -1
        if rotationImage > 0:
            rotatedImage = True
        else:
            rotatedImage = False
        if rotationImage != rotatedAngleOCR and rotatedAngleOCR != 90:
            print(indh,'something has happened with rotation -- page says angle =',
                  rotationImage, 'words say angle =', rotatedAngleOCR)  
        if rotatedAngleOCR > 0: rotatedImage = True # also by words
        rot = rotatedImage
        
        return backtorgb, image_np, rotatedImage, rotatedAngleOCR, bbox_hocr, \
          bboxes_words, bbsq, cbsq, rotation, bbox_par
    
    
# ------------- IMAGE PROCESSING ------------------------
def get_image_process_boxes(backtorgb, bbox_hocr, rotatedImage):
    # search for tags -- this should also be codified in a config file somewhere...
    results_fig_hocr = []
    # we will only do smearing on paragraphs probably, but just doing carea here until we're 100% decided
    blocksImgCar = backtorgb.copy(); blocksImgPar = backtorgb.copy()
    blocksImgCar[:,:,:] = 255; blocksImgPar[:,:,:] = 255
    #del backtorgb
    for (x,y,w,h),text,c,r,bbpar,bbcar in bbox_hocr:
        # fuzzy search
        if (len(text) > 0) and c >= config.ccut_ocr_figcap:    
            if ('high' not in text.strip().lower()): # not 100% sure what this is about...
                if len(text) < config.len_text:
                    if regex.match( '(FIG){e<=1}', text, re.IGNORECASE ):
                        results_fig_hocr.append( ((x,y,w,h), text,c,r,bbpar,bbcar) )
                elif len(text) >= config.len_text1 and len(text) < config.len_text2:
                    if regex.match( '(FIGURE){e<=2}', text, re.IGNORECASE ):
                        results_fig_hocr.append( ((x,y,w,h), text,c,r,bbpar,bbcar) )
                    elif regex.match( '(PLATE){e<=2}', text, re.IGNORECASE ):
                        results_fig_hocr.append( ((x,y,w,h), text,c,r,bbpar,bbcar) )

            # plot each paragraph
            cv.rectangle( blocksImgPar, (bbpar[0], bbpar[1]), 
                         (bbpar[0]+bbpar[2], bbpar[1]+bbpar[3]), (0, 0, 255), -1 )
            # and each carea
            cv.rectangle( blocksImgCar, (bbcar[0],bbcar[1]),
                         (bbcar[0]+bbcar[2],bbcar[1]+bbcar[3]), (0, 0, 255), -1 )
    del blocksImgCar # unless we decide to use

    # draw contours for smeared paragraphs
    kuse = config.kpar
    # if image is rotated, un-rotate
    if rotatedImage: # have rotated image -> rotate smear
        kuse = config.kparrot

    gray = cv.cvtColor(blocksImgPar, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, config.blurKernel, 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # regular
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kuse)
    dilate = cv.dilate(thresh, kernel, iterations=4)
    cnts = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # not sure what this does, but ok

    bbox_figcap_pars1 = []; figcap_rot1 = [] # these will store our bounding boxes from this method
    minVal = [1e5,1e5]
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)
        # does this overlap with a fig tag'd word?
        x1min = x; y1min = y; x1max = x+w; y1max = y+h
        bboxOver = []#; rotOver = []
        #print('--')
        for (x2min,y2min,w2,h2),text,c,r,bbpar,bbcar in results_fig_hocr: 
            x2max = x2min+w2; y2max = y2min+h2
            isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                               (x2min,y2min,x2max,y2max))
            # check that not larger than "fig" tag
            #isOverlapping = (x1min <= x2max and x2min <= x1max and y1min <= y2max and y2min <= y1max)
            #print(x1min,y1min)
            if isOverlapping: # if overlapping -- grab whole SMEARED paragraph
                minVal = [min([x2min,minVal[0]]),min([y2min,minVal[1]])]
                # if not rotatedImage:
                #     #y1min = max([y2min,y1min])
                #     minVal.append(y2min)
                # else:
                #     #x1min = max([x2min,x1min])
                #     minVal.append(x2min)
                #print(x1min,y1min)
                bboxOver.append( (x1min,y1min,x1max,y1max) ) # save rotation for finding "top" later
        # make sure you fix the overlap 
        bboxOver2 = []
        #for (x1min,y1min,x1max,y1max),mv in zip(bboxOver,minVal):
        for x1min,y1min,x1max,y1max in bboxOver:
            if not rotatedImage:
                bboxOver2.append( (x1min,max([y1min,minVal[1]]),x1max,y1max) )
            else:
                bboxOver2.append( (max([x1min,minVal[0]]),y1min,x1max,y1max) )
        bbox_figcap_pars1.append(bboxOver2)
        #print(bboxOver)
        #print(bboxOver2)

    # loop and fill
    bbox_figcap_pars = []; captionText_figcap = []
    for b in bbox_figcap_pars1:
        #print(b)
        if len(b)>0: # have a thing!
            bb = np.unique(b,axis=0) # unique bounding boxes
            if len(bb) > 1: 
                print('we have issue')
                import sys;sys.exit()
            else: 
                bb = bb[0]
            # loop and get all text rotations
            rr = []; captionTextHeur = []
            x1min, y1min, x1max, y1max = bb
            for (x2min,y2min,w,h),text,c,r,bbpar,bbcar in bbox_hocr:
                x2max = x2min+w; y2max = y2min+h
                #isOverlapping = (x1min <= x2max and x2min <= x1max and y1min <= y2max and y2min <= y1max)
                isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                   (x2min,y2min,x2max,y2max))

                if isOverlapping:
                    rr.append(r)
                    captionTextHeur.append( (x2min,y2min,x2max,y2max,text) )
            if len(rr) == 0: rr=[0]
            rrr = stats.mode(rr).mode[0]
            bbox_figcap_pars.append((bb[0],bb[1],bb[2],bb[3],rrr))
            #import sys; sys.exit()
            #if rrr != 0: import sys; sys.exit()
            captionText_figcap.append(captionTextHeur)



    del blur
    del gray
    del thresh
    del dilate
    del kernel
    del blocksImgPar
    
    return captionText_figcap, bbox_figcap_pars



################# CLEAN OVERLAPPING SQUARES #########################
def clean_overlapping_squares(boxes1,scores1,labels1,imgs_name):
    #here I'm going to check for large overlaps and take the largest overlap
    sboxes_cleaned = []; slabels_cleaned = []; sscores_cleaned = []
    bo2 = []; pl2 = []; sc2 = []
    for ib,b in enumerate(boxes1):
        scoreMax = -1
        ioverlap = -1
        for ib2, b2 in enumerate(boxes1):
            if ib2 < ib: # don't double count
                x1min = b[0]; y1min = b[1]; x1max = b[2]; y1max = b[3]
                x2min = b2[0]; y2min = b2[1]; x2max = b2[2]; y2max = b2[3]
                isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                   (x2min,y2min,x2max,y2max))

                w1,h1 = x1max-x1min,y1max-y1min
                x1,y1 = x1min+0.5*w1, y1min+0.5*h1
                w2,h2 = x2max-x2min,y2max-y2min
                x2,y2 = x2min+0.5*w2, y2min+0.5*h2
                iou1 = iou_orig(x1,y1,w1,h1, x2,y2,w2,h2)
                if isOverlapping and iou1 > 0.25: # make sure they are strongly overlapping
                    if scores1[ib2] > scoreMax: 
                        ioverlap = ib2
                        scoreMax = scores1[ib2]
        # final check
        if scores1[ib] > scoreMax: # if the current box has a bigger score, take that one
            ioverlap = ib

        if scores1[ioverlap] == -1: 
            print('issue with scores on', imgs_name)
            import sys; sys.exit()

        bo2.append(boxes1[ioverlap]); pl2.append(labels1[ioverlap]);
        sc2.append(scores1[ioverlap])
    sboxes_cleaned,uind = np.unique(bo2, axis=0, return_index=True) 
    slabels_cleaned = np.array(pl2)[uind]; sscores_cleaned = np.array(sc2)[uind]
    
    return sboxes_cleaned, slabels_cleaned, sscores_cleaned


######################## MERGE PDFSQUARES ################################## 
# use any PDF-mining results to find extra tables and any found captions
# found figures are generally not accurate, so ignore these
def clean_merge_pdfsquares(pdfboxes,pdfrawboxes,sboxes_cleaned, 
                           slabels_cleaned, sscores_cleaned, LABELS, 
                          dfMS):        
    fracx = dfMS['w'].values[0]*1.0/config.IMAGE_W
    fracy = dfMS['h'].values[0]*1.0/config.IMAGE_H  
    if len(pdfboxes) > 0:
        pdfcapboxes = pdfboxes
        if len(pdfrawboxes) > 0:
            pdfcapboxes.extend(pdfrawboxes)
    elif len(pdfrawboxes) > 0:
        pdfcapboxes = pdfrawboxes
    else:
        pdfcapboxes = []
    # mess with
    # if we have somthing, loop and get it
    # for b,l in zip(sboxes_cleaned, slabels_cleaned):
    #     if LABELS[l] == 'figure caption':
    #         print(b,l)
    if len(pdfcapboxes) > 0:
        boxesOut = []; labelsOut = []; scoresOut = []; tagged_pdf_box_ind = []
        # these boxes: xmin,ymin,xmax,ymax -- found by YOLO, IMAGE_W,IMAGE_H max
        for b,l,ss in zip(sboxes_cleaned, slabels_cleaned, sscores_cleaned): 
            # look for negatives
            x1min = max([0,b[0]])*fracx; y1min = max([0,b[1]])*fracy; 
            x1max = min([config.IMAGE_W,b[2]])*fracx; 
            y1max = min([config.IMAGE_H,b[3]])*fracy
            # are we even dealing with a figure caption?
            if 'figure caption' in LABELS[int(l)].lower():
                iouMax = -10; 
                indIou = []
                # bb in xmin,ymin,xmax,ymax in YOLO units -- IMAGE_W, IMAGE_H
                for ibb,bb in enumerate(pdfcapboxes): 
                    # multiply
                    x2min = bb[0] *fracx; y2min=bb[1]*fracy; 
                    x2max=bb[2]*fracx; y2max=bb[3]*fracy; ll = bb[4]
                    isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                       (x2min,y2min,x2max,y2max)) \
                       and (bb[-1] == 'figure caption' or bb[-1] == 'raw')
                    if isOverlapping:
                        if len(indIou) == 0:
                            # take max of overlap w/box
                            #. ..annotations are bigger than captions often
                            indIou = [ min([x2min]),min([y2min]),
                                      max([x2max]),max([y2max]) ] 
                            tagged_pdf_box_ind.append(ibb)
                        else:
                            xo = indIou[0]; yo = indIou[1]; 
                            xo1 = indIou[2]; yo1 = indIou[3]
                            indIou = [ min([xo,x2min]), min([yo,y2min]), 
                                      max([xo1,x2max]), max([yo1,y2max])]
                            tagged_pdf_box_ind.append(ibb) # tag that we found this box
                if len(indIou) > 0: # found 1 that overlapped
                    boxesOut.append((indIou[0]/fracx,indIou[1]/fracy,
                                     indIou[2]/fracx,indIou[3]/fracy))
                    labelsOut.append(l); scoresOut.append(ss)
                else:
                    boxesOut.append([x1min/fracx,y1min/fracy,
                                     x1max/fracx,y1max/fracy]); 
                    #print(x1min,y1min,x1max,y1max)
                    labelsOut.append(l); scoresOut.append(ss)
            elif 'table' in LABELS[int(l)].lower(): # merge into table
                iouMax = -10; 
                indIou = []
                # b in xmin,ymin,xmax,ymax in YOLO units -- IMAGE_W, IMAGE_H
                for ibb,bb in enumerate(pdfcapboxes): 
                    x2min = bb[0] *fracx; y2min=bb[1]*fracy; 
                    x2max=bb[2]*fracx; y2max=bb[3]*fracy; ll = bb[4]
                    isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                       (x2min,y2min,x2max,y2max)) \
                       and (bb[-1] == 'table' or bb[-1] == 'raw')
                    if isOverlapping:
                        if len(indIou) == 0:
                            # take max of overlap w/box
                            # -- annotations are bigger than captions often
                            indIou = [ min([x2min]),min([y2min,y1min]),
                                      max([x2max]),max([y2max,y1max]) ] 
                            tagged_pdf_box_ind.append(ibb)
                        else:
                            xo = indIou[0]; yo = indIou[1]; xo1 = indIou[2]; 
                            yo1 = indIou[3]
                            indIou = [ min([xo,x2min]), min([yo,y2min,y1min]),
                                      max([xo1,x2max]), max([yo1,y2max,y1max])]
                            tagged_pdf_box_ind.append(ibb)
                if len(indIou) > 0: # found 1 that overlapped
                    boxesOut.append((indIou[0]/fracx,indIou[1]/fracy,
                                     indIou[2]/fracx,indIou[3]/fracy))
                    labelsOut.append(l); scoresOut.append(ss)
                else:
                    boxesOut.append([x1min/fracx,y1min/fracy,x1max/fracx,y1max/fracy]); 
                    labelsOut.append(l); scoresOut.append(ss)
            else:
                boxesOut.append([x1min/fracx,y1min/fracy,x1max/fracx,y1max/fracy]); 
                labelsOut.append(l); scoresOut.append(ss)
        # reset
        boxesOut = np.array(boxesOut)
        boxes = boxesOut; labels= labelsOut; scores=scoresOut
        # also, be sure to include PDFbox, and take unique... 
        boxes = boxes.tolist()
        boxes_pdf=boxes; labels_pdf = labels; scores_pdf = scores
    else:
        boxes_pdf = sboxes_cleaned; labels_pdf = slabels_cleaned; 
        scores_pdf = sscores_cleaned
    # #print(boxes_pdf,labels_pdf)
    # print('---')
    # for b,l in zip(boxes_pdf, labels_pdf):
    #     if LABELS[l] == 'figure caption':
    #         print(b,l)
        
    return boxes_pdf, labels_pdf, scores_pdf


#################################### CLEAN BY HEURISTIC FIG CAPTIONS ####################################
def clean_merge_heurstic_captions(boxes_pdf, labels_pdf, scores_pdf, 
                                  bbox_figcap_pars, LABELS,dfMS,
                                 width=None, height=None):
    #for found boxes: get overlap with figure caption from paragraphs 
    boxesOut = []; labelsOut = []; scoresOut = []; pars_in_found_box = []
    ibbOverlap = []
    if dfMS is not None:
        fracx = dfMS['w'].values[0]*1.0/config.IMAGE_W
        fracy = dfMS['h'].values[0]*1.0/config.IMAGE_H  
    elif (width is not None) and (height is not None):
        fracx = width*1.0/config.IMAGE_W
        fracy = height*1.0/config.IMAGE_H  
    else:
        print('no idea...')
        import sys; sys.exit()
    boxes_heur_tf = []
    # these boxes: xmin,ymin,xmax,ymax -- found by YOLO, IMAGE_W,IMAGE_H max
    for b,l,ss in zip(boxes_pdf, labels_pdf, scores_pdf): 
        x1min = max([0,b[0]])*fracx; 
        y1min = max([0,b[1]])*fracy; 
        x1max = min([config.IMAGE_W,b[2]])*fracx; 
        y1max = min([config.IMAGE_H,b[3]])*fracy
        # are we even dealing with a caption?
        bboxOverlap = []; bboxOverlapOrig = []
        if 'caption' in LABELS[int(l)].lower():
            for ibb,bb in enumerate(bbox_figcap_pars):
                x2min, y2min, x2max, y2max,r = bb
                # from x,y,w,h
                # is within... vs...
                if config.found_overlap == 'overlap':
                    isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                       (x2min,y2min,x2max,y2max))
                elif config.found_overlap == 'center':
                    # center is overlapping
                    x2 = 0.5*(x2min+x2max); y2 = 0.5*(y2min+y2max)
                    isOverlapping = (x2 <= x1max) and (x2 >= x1min) and (y2 <= y1max) and (y2 >= y1min)

                if isOverlapping: # if its overlapping, take the heuristic bounding box
                    if r == 0:
                        ymin = y2min
                        xmin = min([x1min,x2min]); 
                        xmax = max([x1max,x2max]); 
                        ymax = max([y1max,y2max])
                    else: # assume 90?
                        xmin = x2min
                        ymin = min([y1min,y2min]); 
                        xmax = max([x1max,x2max]); 
                        ymax = max([y1max,y2max])
                    bboxOverlap.append((xmin,ymin,xmax,ymax))
                    ibbOverlap.append(ibb)
            #print(np.shape(bboxOverlap)[0])
            if np.shape(bboxOverlap)[0]==1: # found 1 that overlapped
                boxesOut.append((bboxOverlap[0][0]/fracx,bboxOverlap[0][1]/fracy,
                                 bboxOverlap[0][2]/fracx,bboxOverlap[0][3]/fracy))
                labelsOut.append(l); scoresOut.append(ss)
                boxes_heur_tf.append(True)
            elif np.shape(bboxOverlap)[0]>1: # found more than 1 that overlapped
                # pick best overlap with IOU
                ious1 = []
                #ibbOverlap2 = []
                #**HERE SOMEWHERE IS THE ISSUE**
                for ibb1,bb1 in enumerate(bboxOverlap):
                    w1,h1 = bb1[2]-bb1[0],bb1[3]-bb1[1]
                    x1,y1 = bb1[0]+0.5*w1, bb1[1]+0.5*h1
                    iouMax2 = -1000
                    for ibb,bb in enumerate(bbox_figcap_pars):
                    #if ibb in ibbOverlap:
                        x2min, y2min, x2max, y2max,r = bb
                        w2,h2 = x2max-x2min,y2max-y2min
                        x2,y2 = x2min+0.5*w2, y2min+0.5*h2
                        iouhere = iou_orig(x1,y1,w1,h1, x2,y2,w2,h2)
                        if iouhere > iouMax2:
                            iouMax2 = iouhere
                            #ious1.append()
                    ious1.append(iouMax2)
                # which is max IOU
                indMax = np.argmax(ious1)
                #print(bboxOverlap)
                #print(indMax)
                #print(ibbOverlap)
                boxesOut.append((bboxOverlap[indMax][0]/fracx,
                                 bboxOverlap[indMax][1]/fracy,
                                 bboxOverlap[indMax][2]/fracx,
                                 bboxOverlap[indMax][3]/fracy))
                labelsOut.append(l); scoresOut.append(ss)
                boxes_heur_tf.append(True)
                #remove others
                #for ibb in range(len(ibbOverlap)):
                #    if ibb != indMax:
                #        p = ibbOverlap.pop
                #print(bboxOverlap)
                
            else:
                boxesOut.append([x1min/fracx,y1min/fracy,x1max/fracx,y1max/fracy]); 
                labelsOut.append(l); scoresOut.append(ss)
                boxes_heur_tf.append(False)
        else:
            boxesOut.append([x1min/fracx,y1min/fracy,x1max/fracx,y1max/fracy]); 
            labelsOut.append(l); scoresOut.append(ss)
            boxes_heur_tf.append(False)
    # reset
    boxesOut = np.array(boxesOut)
    boxes_heur = boxesOut; labels_heur = labelsOut; scores_heur=scoresOut 
    
    return boxes_heur, labels_heur, scores_heur, ibbOverlap, boxes_heur_tf


        ##################### LOOK FOR HEURISTIC CAPTIONS NOT OTHERWISE FOUND #################
def add_heuristic_captions(bbox_figcap_pars,captionText_figcap,ibbOverlap,
                           boxes_heur_in, labels_heur_in, scores_heur_in, dfMS):
    fracx = dfMS['w'].values[0]*1.0/config.IMAGE_W
    fracy = dfMS['h'].values[0]*1.0/config.IMAGE_H  
    # regex search terms for fuzzy search
    # how long in words should each caption be for each keyword?
    capInd = [] # save index of caption block tagged
    for (ibb,bb),caps in zip(enumerate(bbox_figcap_pars), captionText_figcap): # loop over each potential caption block 
        lenCap = len(caps) # how many words in whole block?
        if ibb not in ibbOverlap: # not already counted
            for icap,cap in enumerate(caps):
                for ik,k in enumerate(config.keyWords): # for each fuzzy search, in order
                    if icap < config.lookLength and lenCap >= config.lenMin[ik]: # only look at starting words of block, make sure block is long enough to be figure
                        if regex.match( k, cap[-1], re.IGNORECASE ):
                            capInd.append(ibb)

    boxes_heur = boxes_heur_in.tolist().copy(); 
    labels_heur = labels_heur_in.copy()
    scores_heur = scores_heur_in.copy()
    if len(capInd) > 0:
        capInd = np.unique(capInd) # double count if multiple keywords found
        for ibb,bb in enumerate(bbox_figcap_pars):
            if ibb in capInd:
                boxes_heur.append((bb[0]/fracx,bb[1]/fracy,bb[2]/fracx,bb[3]/fracy)); 
                #labels_heur.append(LABELS.index('figure caption')); 
                # special label for these last ones
                labels_heur.append(-2); 
                scores_heur.append(1.0) # fake score for now
    boxes_heur = np.array(boxes_heur)  
    
    return boxes_heur, labels_heur, scores_heur

########### CLEAN BOTH PREDICTED AND TRUE BOXES BY OVERLAP WITH PARAGRAPHS ############
def clean_found_overlap_with_ocr(boxes_heur, labels_heur, scores_heur,bboxes_words,
                                 bbox_par,rotation,LABELS, dfMS, boxes_heur_tf, 
                                 width=None, height=None):
    if dfMS is not None:
        fracx = dfMS['w'].values[0]*1.0/config.IMAGE_W
        fracy = dfMS['h'].values[0]*1.0/config.IMAGE_H  
    elif (width is not None) and (height is not None):
        fracx = width*1.0/config.IMAGE_W
        fracy = height*1.0/config.IMAGE_H  
    else:
        print('WHAT')
        import sys; sys.exit()
        
    # combine paragraphs & words
    bboxes_combined = []
    for ibb, bp in enumerate(bboxes_words): 
        bb,texts,confs = bp 
        bboxes_combined.append(bb)
    for ibb, bp in enumerate(bbox_par): # these are also xmin,ymin,xmax,ymax -- found w/OCR, original page size
        bb,aa,ll = bp    
        bboxes_combined.append(bb)

    #for found boxes: get overlap with figure caption from paragraphs 
    boxesOut = []; labelsOut = []; scoresOut = []; pars_in_found_box = []
    for b,l,ss,bhtf in zip(boxes_heur, labels_heur, scores_heur, boxes_heur_tf): # these boxes: xmin,ymin,xmax,ymax -- found by YOLO, IMAGE_W,IMAGE_H max
        # look for negatives
        b[0] = max([0,b[0]]); b[1] = max([0,b[1]]); 
        b[2] = min([config.IMAGE_W,b[2]]); 
        b[3] = min([config.IMAGE_H,b[3]])
        x1min = b[0]*fracx; y1min = b[1]*fracy; x1max = b[2]*fracx; y1max = b[3]*fracy
        # are we even dealing with a caption?
        if 'caption' in LABELS[int(l)].lower() or l == -2: # include special label for heuristic-only caption
            indIou = [1e10,1e10,-1e10,-1e10]
            indIou2 = indIou.copy()#; indIou2[0] = 2e10; iou1 = 1e10
            indIou2[0] *= 2; indIou2[1] *= 2; indIou2[2] *= 2; indIou2[3] *= 2
            # don't expand super far in y direction, only x
            i10 = indIou[0]; i11=indIou[2]; i20 = indIou2[0]; i21 = indIou2[2]
            rot = 0
            if len(rotation)>0: rot = stats.mode(rotation).mode[0]
            if rot != 90:
                i10 = indIou[1]; i11 = indIou[3]; i20 = indIou2[1]; i21 = indIou2[2]
            while (i10 != i20) and (i11 != i21):
                indIou2 = indIou
                i10 = indIou[0]; i11=indIou[2]; i20 = indIou2[0]; i21 = indIou2[2]
                if rot != 90:
                    i10 = indIou[1]; i11 = indIou[3]; i20 = indIou2[1]; i21 = indIou2[2]
                for ibb,bb in enumerate(bboxes_combined):
                    x2min, y2min, x2max, y2max = bb
                    # is within... vs...
                    if config.found_overlap == 'overlap':
                        isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                           (x2min,y2min,x2max,y2max))
                    elif config.found_overlap == 'center':
                        # center is overlapping
                        x2 = 0.5*(x2min+x2max); y2 = 0.5*(y2min+y2max)
                        isOverlapping = (x2 <= x1max) and (x2 >= x1min) and (y2 <= y1max) and (y2 >= y1min)
                    if isOverlapping:# and iou1 <= 1.0:
                        xo = indIou[0]; yo = indIou[1]; xo1 = indIou[2]; yo1 = indIou[3]
                        indIou = [ min([xo,bb[0]]), min([yo,bb[1]]), 
                                  max([xo1,bb[2]]), max([yo1,bb[3]])]
                if indIou[0] != 1e10: # found 1 that overlapped
                    if rot == 90: # right-side up
                        x1min, x1max = indIou[0],indIou[2]
                    else:
                        y1min, y1max = indIou[1],indIou[3]

            if indIou[0] != 1e10: # found 1 that overlapped
                myboxhere1 = [indIou[0]/fracx,indIou[1]/fracy,indIou[2]/fracx,indIou[3]/fracy]
                # if overlap with a heuristic box,take that max
                myboxhere = []
                if bhtf: # yes, a heuristic box -- take closes to "Fig"
                    if rot == 0: # not rotated
                        myboxhere = [myboxhere1[0],max([myboxhere1[1],y1min/fracy]), 
                                     myboxhere1[2],myboxhere1[3]]
                    else:
                        myboxhere = [max([myboxhere1[0],x1min/fracx]),myboxhere1[1], 
                                     myboxhere1[2],myboxhere1[3]]
                else:
                    myboxhere = myboxhere1
                        
                #print('found overlap', indIou)
                boxesOut.append(myboxhere)
                labelsOut.append(l); scoresOut.append(ss)
            else:
                boxesOut.append(b); labelsOut.append(l); scoresOut.append(ss)
        else:
            boxesOut.append(b); labelsOut.append(l); scoresOut.append(ss)
    # reset
    boxesOut = np.array(boxesOut)
    boxes_par_found = boxesOut; labels_par_found = labelsOut; scores_par_found=scoresOut
    return boxes_par_found, labels_par_found, scores_par_found


def clean_true_overlap_with_ocr(truebox, bboxes_words,bbox_par,rotation, LABELS, dfMS):
    fracx = dfMS['w'].values[0]*1.0/config.IMAGE_W
    fracy = dfMS['h'].values[0]*1.0/config.IMAGE_H  
    bboxes_combined = []
    for ibb, bp in enumerate(bboxes_words): 
        bb,texts,confs = bp 
        bboxes_combined.append(bb)
    for ibb, bp in enumerate(bbox_par): # these are also xmin,ymin,xmax,ymax -- found w/OCR, original page size
        bb,aa,ll = bp    
        bboxes_combined.append(bb)

    #for annotated boxes: ALSO get overlap with figure caption from WORDS
    # --> annotated boxes can also be much larger than the words so make sure all things are INSIDE
    trueBoxOut = []; pars_in_true_box = []
    # find figure boxes so that we don't overlap with them
    fig_boxes = []
    for b in truebox:
        if LABELS[int(b[4]-1)] == 'figure':
            fig_boxes.append(b)
    rot = 0
    if len(rotation)>0: rot = stats.mode(rotation).mode[0]

    # loop and update annotations        
    for b in truebox: # these boxes: xmin,ymin,xmax,ymax -- found by YOLO, IMAGE_W,IMAGE_H max
        l = LABELS[int(b[4]-1)]
        # look for negatives -- not really necessary here
        b[0] = max([0,b[0]]); b[1] = max([0,b[1]]); 
        b[2] = min([config.IMAGE_W,b[2]]); 
        b[3] = min([config.IMAGE_H,b[3]])
        # are we even dealing with a caption?
        if 'caption' in l.lower():
            x1min = b[0]*fracx; y1min = b[1]*fracy; x1max = b[2]*fracx; y1max = b[3]*fracy # inside?
            indIou = [1e10,1e10,-1e10,-1e10]
            indIou2 = indIou.copy(); #indIou2[0] = 2e10
            indIou2[0] *= 2; indIou2[1] *= 2; indIou2[2] *= 2; indIou2[3] *= 2
            # don't expand super far in y direction, only x
            i10 = indIou[0]; i11=indIou[2]; i20 = indIou2[0]; i21 = indIou2[2]
            if rot != 90:
                i10 = indIou[1]; i11 = indIou[3]; i20 = indIou2[1]; i21 = indIou2[2]
            while (i10 != i20) and (i11 != i21):
                indIou2 = indIou
                i10 = indIou[0]; i11=indIou[2]; i20 = indIou2[0]; i21 = indIou2[2]
                if rot != 90:
                    i10 = indIou[1]; i11 = indIou[3]; i20 = indIou2[1]; i21 = indIou2[2]
                for ibb,bb in enumerate(bboxes_combined):
                    x2min, y2min, x2max, y2max = bb
                    # is within....vs...
                    if config.true_overlap == 'overlap':
                        isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                           (x2min,y2min,x2max,y2max))

                        #center is within
                    elif config.true_overlap == 'center':
                        x2 = 0.5*(x2min+x2max); y2 = 0.5*(y2min+y2max)
                        isOverlapping = (x2 <= x1max) and (x2 >= x1min) and (y2 <= y1max) and (y2 >= y1min)
                    # using whichever condition -- change box sizes
                    if isOverlapping:
                        xo = indIou[0]; yo = indIou[1]; xo1 = indIou[2]; yo1 = indIou[3]
                        indIou = [ min([xo,bb[0]]), min([yo,bb[1]]), 
                                  max([xo1,bb[2]]), max([yo1,bb[3]])]
                if indIou[0] != 1e10: # found 1 that overlapped
                    if stats.mode(rotation).mode[0] == 90: # right-side up
                        x1min, x1max = indIou[0],indIou[2]
                    else:
                        y1min, y1max = indIou[1],indIou[3]            
  
            if indIou[0] != 1e10: # found 1 that overlapped
                trueBoxOut.append((indIou[0]/fracx,indIou[1]/fracy,indIou[2]/fracx,indIou[3]/fracy, b[4]))
            else:
                trueBoxOut.append(b)
        ##### come back and do table!!! ####
#         elif 'table' in l.lower():
#             iouMax = -10; 
#             indIou = []
#             #for ibb, bp in enumerate(bboxes_words): # these are also xmin,ymin,xmax,ymax -- found w/OCR, original page size
#             #    bb,texts,confs = bp
#             for ibb, bp in enumerate(bbox_par): # these are also xmin,ymin,xmax,ymax -- found w/OCR, original page size
#                 bb,aa,ll = bp
#                 #x1min = b[0]*fracx; y1min = b[1]*fracy; x1max = b[2]*fracx; y1max = b[3]*fracy
#                 x2min, y2min, x2max, y2max = bb
#                 #isOverlapping = (x1min <= x2max and x2min <= x1max and y1min <= y2max and y2min <= y1max)
#                 x2 = 0.5*(x2min+x2max); y2 = 0.5*(y2min+y2max)
#                 isOverlapping = (x2 <= x1max) and (x2 >= x1min) and (y2 <= y1max) and (y2 >= y1min)
#                 if isOverlapping:
#                     if len(indIou) == 0:
#                         #indIou = [ min([bb[0],x1min]),min([bb[1],y1min]),
#                         #          max([bb[2],x1max]),max([bb[3],y1max]) ] 
#                         indIou = [bb[0],bb[1],bb[2],bb[3]]
#                     else:
#                         xo = indIou[0]; yo = indIou[1]; xo1 = indIou[2]; yo1 = indIou[3]
#                         #indIou = [ min([xo,bb[0],x1min]), min([yo,bb[1],y1min]), 
#                         #          max([xo1,bb[2],x1max]), max([yo1,bb[3],y1max])]
#                         indIou = [ min([xo,bb[0]]), min([yo,bb[1]]), 
#                                   max([xo1,bb[2]]), max([yo1,bb[3]])]                    
#             if len(indIou) > 0: # found 1 that overlapped
#                 trueBoxOut.append((indIou[0]/fracx,indIou[1]/fracy,indIou[2]/fracx,indIou[3]/fracy, b[4]))
#             else:
#                 trueBoxOut.append(b)
        else:
            trueBoxOut.append(b)
    # reset true box
    truebox1 = np.array(trueBoxOut)
    return truebox1

        ########### CLEAN FIGURES BY CHECKING IF THEY OVERLAP WITH SQUARES -- IF NO FIG, TAKE SQUARE ############
def clean_merge_squares(bbsq_in, cbsq, boxes_par_found, labels_par_found, 
                        scores_par_found, LABELS, dfMS, useColorbars=False, 
                       width=None, height=None):
    if dfMS is not None:
        fracx = dfMS['w'].values[0]*1.0/config.IMAGE_W
        fracy = dfMS['h'].values[0]*1.0/config.IMAGE_H  
    elif (width is not None) and (height is not None):
        fracx = width*1.0/config.IMAGE_W
        fracy = height*1.0/config.IMAGE_H  
        
    boxesOut = []; labelsOut = []; scoresOut = []
    bbsq = bbsq_in.copy()
    if useColorbars:
        for ss in cbsq:
            bbsq.append(ss)
    # these boxes: xmin,ymin,xmax,ymax -- found by YOLO, IMAGE_W,IMAGE_H max
    for b,l,ss in zip(boxes_par_found, labels_par_found, scores_par_found): 
        # look for negatives
        b[0] = max([0,b[0]]); b[1] = max([0,b[1]]); 
        b[2] = min([config.IMAGE_W,b[2]]); 
        b[3] = min([config.IMAGE_H,b[3]])
        x1min = b[0]*fracx; y1min = b[1]*fracy; x1max = b[2]*fracx; y1max = b[3]*fracy
        # are we even dealing with a caption?
        if 'figure' == LABELS[int(l)].lower():
            indIou = [1e10,1e10,-1e10,-1e10]
            indIou2 = indIou.copy(); indIou2[0] = 2e10; iou1 = 1e10
            for ibb,bb in enumerate(bbsq): # only do the once
                x2min, y2min, x2max, y2max = bb
                isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                   (x2min,y2min,x2max,y2max))
                if isOverlapping:
                    xo = indIou[0]; yo = indIou[1]; xo1 = indIou[2]; yo1 = indIou[3]
                    indIou = [ min([xo,bb[0],x1min]), min([yo,bb[1],y1min]), 
                              max([xo1,bb[2],x1max]), max([yo1,bb[3],y1max])]

            if indIou[0] != 1e10: # found 1 that overlapped
                boxesOut.append((indIou[0]/fracx,indIou[1]/fracy,indIou[2]/fracx,indIou[3]/fracy))
                labelsOut.append(l); scoresOut.append(ss)
            else:
                boxesOut.append(b); labelsOut.append(l); scoresOut.append(ss)
        else:
            boxesOut.append(b); labelsOut.append(l); scoresOut.append(ss)
    # reset
    boxesOut = np.array(boxesOut)
    # replace
    boxes_sq1 = boxesOut; labels_sq1 = labelsOut; scores_sq1 = scoresOut
    return boxes_sq1, labels_sq1, scores_sq1, bbsq



################# CLEAN OBVIOUSLY BIG CAPTIONS ########################
def clean_big_captions(boxes_sq1,labels_sq1,scores_sq1, LABELS):
    boxes_sq = []; labels_sq = []; scores_sq = []
    for b,l,s in zip(boxes_sq1,labels_sq1,scores_sq1):
        area = (b[2]-b[0])*(b[3]-b[1])/(1.0*config.IMAGE_W*config.IMAGE_H)
        # if small area OR not a figure caption
        if area <= 0.75 or ((LABELS[int(l)] != 'figure caption' and l!=-2) or (l==-2)):
            boxes_sq.append(b); labels_sq.append(l); scores_sq.append(s) 
    return boxes_sq, labels_sq, scores_sq


############### MATCH SQUARES TO CAPTIONS #############################
def clean_match_fig_cap(boxes_sq,labels_sq,scores_sq, bbsq, 
                        LABELS, rotatedImage, 
                        rotatedAngleOCR,dfMS,change_bottoms=False, 
                       width=None, height=None):
    if dfMS is not None:
        fracx = dfMS['w'].values[0]*1.0/config.IMAGE_W
        fracy = dfMS['h'].values[0]*1.0/config.IMAGE_H  
    elif (width is not None) and (height is not None):
        fracx = width*1.0/config.IMAGE_W
        fracy = height*1.0/config.IMAGE_H  
        
    # take caption closest to the BOTTOM EDGE, NOT in a square
    boxesOut = []; labelsOut = []; scoresOut = []
    boxes_fig = []; boxes_cap = []; boxes_heur_cap = []
    for b,l,s in zip(boxes_sq,labels_sq,scores_sq):
        if LABELS[int(l)] == 'figure caption' and l!=-2:
            boxes_cap.append((b,l,s))
            # take out overlaps with image-processing squares here...
        elif LABELS[int(l)] == 'figure' and l!=-2:
            boxes_fig.append((b,l,s))
        elif l == -2:
            boxes_heur_cap.append((b,l,s))
        else:
            boxesOut.append(b)
            scoresOut.append(s)
            labelsOut.append(l)

    icsave = []; icheursave = []; fig_cap_pair_reg = []; fig_cap_pair_heur = []
    # find closest captions
    for ibb,bb in enumerate(boxes_fig):
        mind = 5e15; iout = -1
        xc,yc = 0.5*(bb[0][0]+bb[0][2]),bb[0][3]
        if rotatedImage:
            xc, yc = bb[0][2], 0.5*(bb[0][1]+bb[0][3])
        for ic,bc in enumerate(boxes_cap): # find closest to bottom, not inside a square
            xcc,ycc= 0.5*(bc[0][0]+bc[0][2]),0.5*(bc[0][1]+bc[0][3])
            d = np.sum(((xc-xcc)**2 + (yc-ycc)**2)**0.5)
            if d < mind:
                mind = d
                iout = ic
        if iout > -1: # if we found a mega-yolo-found caption, add it
            icsave.append(iout)
            fig_cap_pair_reg.append([ibb,iout])
        else: # let's try heuristically found ones
            #print('nope')
            mind = 5e15; iout = -1
            xc,yc = 0.5*(bb[0][0]+bb[0][2]),bb[0][3]
            if rotatedImage:
                xc, yc = bb[0][2], 0.5*(bb[0][1]+bb[0][3])
            for ic,bc in enumerate(boxes_heur_cap): # find closest to bottom, not inside a square
                xcc,ycc= 0.5*(bc[0][0]+bc[0][2]),0.5*(bc[0][1]+bc[0][3])
                d = np.sum(((xc-xcc)**2 + (yc-ycc)**2)**0.5)
                if d < mind:
                    mind = d
                    iout = ic
            if iout > -1: # if we found a mega-yolo-found caption, add it
                icheursave.append(iout)
                fig_cap_pair_heur.append([ibb,iout])
            else:
                fig_cap_pair_heur.append([ibb,-1]) # nothing found

    # check for overlaps
    boxesOut2 = []; labelsOut2 = []; scoresOut2 = []
    for f in fig_cap_pair_reg:
        bf = boxes_fig[f[0]][0] # fig box
        bc = boxes_cap[f[1]][0] # cap box
        # add box
        boxesOut2.append(bc)
        labelsOut2.append(boxes_cap[f[1]][1])
        scoresOut2.append(boxes_cap[f[1]][2])
        isOverlapping = isRectangleOverlap(bf,bc)

        if isOverlapping and change_bottoms:
            # which side is overlapping?
            xcf,ycf = 0.5*(bf[0]+bf[2]), 0.5*(bf[1]+bf[3])
            xcc,ycc = 0.5*(bc[0]+bc[2]), 0.5*(bc[1]+bc[3])
            if ycf < ycc and rotatedAngleOCR == 0: # caption is on bottom
                bf[3] = min([bf[3],bc[1]])
                #print('yes on bottom')
            elif xcf < xcc: # caption on right
                bf[2] = min([bf[2],bc[0]])
        boxesOut2.append(bf); labelsOut2.append(boxes_fig[f[0]][1])
        scoresOut2.append(boxes_fig[f[0]][2])

    # check for overlaps
    for f in fig_cap_pair_heur:
        if f[1] != -1: # have a associated caption
            bf = boxes_fig[f[0]][0] # fig box
            bc = boxes_heur_cap[f[1]][0] # cap box
            boxesOut2.append(bc)
            labelsOut2.append(LABELS.index('figure caption'))
            scoresOut2.append(boxes_heur_cap[f[1]][2])
            isOverlapping = isRectangleOverlap(bf,bc)

            if isOverlapping and change_bottoms:
                # which side is overlapping?
                xcf,ycf = 0.5*(bf[0]+bf[2]), 0.5*(bf[1]+bf[3])
                xcc,ycc = 0.5*(bc[0]+bc[2]), 0.5*(bc[1]+bc[3])
                if ycf < ycc and rotatedAngleOCR == 0: # caption is on bottom
                    bf[3] = min([bf[3],bc[1]])
                elif xcf < xcc: # caption on right
                    bf[2] = min([bf[2],bc[0]])
            boxesOut2.append(bf); labelsOut2.append(boxes_fig[f[0]][1])
            scoresOut2.append(boxes_fig[f[0]][2])
        else: # just a square, no caption
            bf = boxes_fig[f[0]]
            boxesOut2.append(bf[0])
            labelsOut2.append(bf[1])
            scoresOut2.append(bf[2])

    for b,l,s in zip(boxesOut2,labelsOut2,scoresOut2):
        boxesOut.append(b)
        labelsOut.append(l)
        scoresOut.append(s)

    # # alright, this is silly that we have to do this again
    # # sigh... expand by boxes
    # for ibox in range(len(boxesOut)):
    #     x1min,y1min,x1max,y1max = boxesOut[ibox]
    #     if labelsOut[ibox] == LABELS.index('figure'):
    #         for ibb,bb in enumerate(bbsq): # only do the once
    #             x2min, y2min, x2max, y2max = bb[0]/fracx, bb[1]/fracy, bb[2]/fracx, bb[3]/fracy
    #             #isOverlapping = (x1min <= x2max and x2min <= x1max and y1min <= y2max and y2min <= y1max)
    #             isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
    #                                                (x2min,y2min,x2max,y2max))
    #             if isOverlapping:
    #                 boxesOut[ibox] = (min(x1min,x2min),min(y1min,y2min), 
    #                                   max(x1max,x2max), max(y1max,y2max))           

    # alright, this is silly that we have to do this again
    # sigh... expand by boxes
    for ibox in range(len(boxesOut)):
        x1min,y1min,x1max,y1max = boxesOut[ibox]
        if labelsOut[ibox] == LABELS.index('figure'):
            for ibb,bb in enumerate(bbsq): # only do the once
                x2min, y2min, x2max, y2max = bb[0]/fracx, bb[1]/fracy, bb[2]/fracx, bb[3]/fracy
                isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),(x2min,y2min,x2max,y2max))
                if isOverlapping:
                    boxesOut[ibox] = (min(x1min,x2min),min(y1min,y2min), max(x1max,x2max), max(y1max,y2max))       

    boxes_sq = boxesOut; labels_sq = labelsOut; scores_sq = scoresOut   
    return boxes_sq, labels_sq, scores_sq


########### EXPAND AROUND CAPTIONS (FOUND & ANNOTATIONS) ###############
def expand_true_boxes_fig_cap(truebox1, rotatedImage, LABELS):
    boxes_true_fig = []; boxes_true_cap = []; true_others = []
    fig_cap_pair_true = []
    boxesOutTrue = []
    for it,tbox in enumerate(truebox1):
        if LABELS[int(tbox[-1]-1)] == 'figure caption':
            boxes_true_cap.append(tbox.copy())
        elif LABELS[int(tbox[-1]-1)] == 'figure':
            boxes_true_fig.append(tbox.copy())
        else:
            true_others.append(tbox.copy())

    # pair
    for ibb,bb in enumerate(boxes_true_fig):
        mind = 5e15; iout = -1
        xc,yc = 0.5*(bb[0]+bb[2]),bb[3]
        if rotatedImage:
            xc, yc = bb[2], 0.5*(bb[1]+bb[3])
        for ic,bc in enumerate(boxes_true_cap): # find closest to bottom, not inside a square
            xcc,ycc= 0.5*(bc[0]+bc[2]),0.5*(bc[1]+bc[3])
            d = np.sum(((xc-xcc)**2 + (yc-ycc)**2)**0.5)
            if d < mind:
                mind = d
                iout = ic
        if iout > -1: # if we found a mega-yolo-found caption, add it
            fig_cap_pair_true.append([ibb,iout])
        else: # let's try heuristically found ones
            fig_cap_pair_true.append([ibb,-1]) # nothing found

    # expand fig->caption for overlaps
    # NOTE: this associates each figure with a caption -- i.e., no "floating" captions
    # floating figures are OK however
    for f in fig_cap_pair_true:
        bf = boxes_true_fig[f[0]].copy() # fig box
        if f[1] != -1: # have a associated caption
            bc = boxes_true_cap[f[1]].copy() # cap box
            boxesOutTrue.append(bc)     
            # which side is caption on?
            xcf,ycf = 0.5*(bf[0]+bf[2]), 0.5*(bf[1]+bf[3])
            xcc,ycc = 0.5*(bc[0]+bc[2]), 0.5*(bc[1]+bc[3])
            xShift = (xcc-xcf)/config.IMAGE_W; 
            yShift = -1.0*(ycc-ycf)/config.IMAGE_H # -1 because y decreases down
            theta = np.arctan2(yShift,xShift)*180/np.pi
            # arctan2 in range [-180,180]
            if (theta < -70) and (theta > -110): # caption on bottom
                bf[3] = bc[1]
            elif (theta < 20) and (theta > -20): # caption on right
                bf[2] = bc[0]
            elif (theta > 160) or (theta < -160):
                bf[0] = bc[2] 
            #import sys; sys.exit()
        boxesOutTrue.append(bf) # either way, add it on
    # tack on others as well
    for tbox in true_others:
        boxesOutTrue.append(tbox.copy())

    truebox1 = boxesOutTrue.copy() # replace
    return truebox1







def expand_found_boxes_fig_cap(boxes_sq, labels_sq, scores_sq, bbsq, 
                               rotatedImage, LABELS, dfMS, 
                              width=None, height=None):
    if dfMS is not None:
        fracx = dfMS['w'].values[0]*1.0/config.IMAGE_W
        fracy = dfMS['h'].values[0]*1.0/config.IMAGE_H  
    elif (width is not None) and (height is not None):
        fracx = width*1.0/config.IMAGE_W
        fracy = height*1.0/config.IMAGE_H  
    # now, do for found boxes
    # Note: in theory, you can do this in the step above, but here I'm just doing them together 
    # for clarity (true vs found)
    boxes_fig = []; boxes_cap = []; boxes_others = []
    labels_fig = []; labels_cap = []; labels_other = []
    scores_fig = []; scores_cap = []; scores_other = []
    fig_cap_pair = []
    boxesOut = []; labelsOut = []; scoresOut = []
    for box,l,s in zip(boxes_sq, labels_sq, scores_sq):
        if LABELS[int(l)] == 'figure caption':
            boxes_cap.append(box)
            labels_cap.append(l); scores_cap.append(s)
        elif LABELS[int(l)] == 'figure':
            boxes_fig.append(box)
            labels_fig.append(l); scores_fig.append(s)
        else:
            boxes_others.append(box)
            labels_other.append(l); scores_other.append(s)

    # pair
    for ibb,bb in enumerate(boxes_fig):
        mind = 5e15; iout = -1
        xc,yc = 0.5*(bb[0]+bb[2]),bb[3]
        if rotatedImage:
            xc, yc = bb[2], 0.5*(bb[1]+bb[3])
        for ic,bc in enumerate(boxes_cap): # find closest to bottom, not inside a square
            xcc,ycc= 0.5*(bc[0]+bc[2]),0.5*(bc[1]+bc[3])
            d = np.sum(((xc-xcc)**2 + (yc-ycc)**2)**0.5)
            if d < mind:
                mind = d
                iout = ic
        if iout > -1: # if we found a mega-yolo-found caption, add it
            fig_cap_pair.append([ibb,iout])
        else: # if no, mark as no caption
            fig_cap_pair.append([ibb,-1]) # nothing found
            
    #print(fig_cap_pair)
    #print('boxes_fig',boxes_fig)
    #print('boxes_cap',boxes_cap)

    # expand fig->caption for overlaps
    # NOTE: this associates each figure with a caption -- i.e., no "floating" captions
    # floating figures are OK however
    for f in fig_cap_pair:
        bf = np.array(boxes_fig[f[0]]) # fig box
        if f[1] != -1: # have a associated caption
            bc = boxes_cap[f[1]].copy() # cap box
            boxesOut.append(bc) # either way, add it on
            labelsOut.append(labels_cap[f[1]])
            scoresOut.append(scores_cap[f[1]])
            # which side is caption on?
            xcf,ycf = 0.5*(bf[0]+bf[2]), 0.5*(bf[1]+bf[3])
            xcc,ycc = 0.5*(bc[0]+bc[2]), 0.5*(bc[1]+bc[3])
            xShift = (xcc-xcf)/config.IMAGE_W; 
            yShift = -1.0*(ycc-ycf)/config.IMAGE_H
            theta = np.arctan2(yShift,xShift)*180/np.pi
            #print(theta)
            # arctan2 in range [-180,180]
            if (theta < -70) and (theta > -110): # caption on bottom
                bf[3] = bc[1]
            elif (theta < 20) and (theta > -20): # caption on right
                bf[2] = bc[0]
            elif (theta > 160) or (theta < -160):
                bf[0] = bc[2] 
            # if (theta < -70) and (theta > -110): # caption on bottom
            #     bf[0] = min(bc[0],bf[0]); bf[2] = max(bf[2],bc[2])
            # elif (theta < 20) and (theta > -20): # caption on right
            #     bf[1]=min(bc[1],bf[1]); bf[3]=max(bf[3],bc[3])
            # elif (theta > 160) or (theta < -160):
            #     bf[1]=min(bc[1],bf[1]); bf[3]=max(bf[3],bc[3])
        boxesOut.append(bf) # either way, add it on
        labelsOut.append(labels_fig[f[0]])
        scoresOut.append(scores_fig[f[0]])
    # for others
    for b,l,s in zip(boxes_others, labels_other, scores_other):
        boxesOut.append(b); labelsOut.append(l); 
        scoresOut.append(s)
        
    #print(boxesOut,labelsOut,scoresOut)

    # yet again we have to do this... can we do this just once here?
    for ibox in range(len(boxesOut)):
        x1min,y1min,x1max,y1max = boxesOut[ibox]
        if labelsOut[ibox] == LABELS.index('figure'):
            for ibb,bb in enumerate(bbsq): # only do the once
                x2min, y2min, x2max, y2max = bb[0]/fracx, bb[1]/fracy, bb[2]/fracx, bb[3]/fracy
                isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                   (x2min,y2min,x2max,y2max))
                if isOverlapping:
                    boxesOut[ibox] = (min(x1min,x2min),min(y1min,y2min), 
                                      max(x1max,x2max), max(y1max,y2max))       
    boxes_sq = boxesOut.copy(); labels_sq = labelsOut.copy(); scores_sq = scoresOut.copy()
    return boxes_sq, labels_sq, scores_sq



def expand_true_area_above_cap(truebox1, rotatedImage, LABELS):
    # so, it might be a little silly that we are doing this again, BUT maybe its not
    boxes_true_fig = []; boxes_true_cap = []; true_others = []
    fig_cap_pair_true = []
    boxesOutTrue = []
    # also, save fig+cap combos
    ###boxesCombTrue = []; boxesCombFound = []
    for it,tbox in enumerate(truebox1):
        if LABELS[int(tbox[-1]-1)] == 'figure caption':
            boxes_true_cap.append(tbox.copy())
        elif LABELS[int(tbox[-1]-1)] == 'figure':
            boxes_true_fig.append(tbox.copy())
        else:
            true_others.append(tbox.copy())

    # pair
    for ibb,bb in enumerate(boxes_true_fig):
        mind = 5e15; iout = -1
        xc,yc = 0.5*(bb[0]+bb[2]),bb[3]
        if rotatedImage:
            xc, yc = bb[2], 0.5*(bb[1]+bb[3])
        for ic,bc in enumerate(boxes_true_cap): # find closest to bottom, not inside a square
            xcc,ycc= 0.5*(bc[0]+bc[2]),0.5*(bc[1]+bc[3])
            d = np.sum(((xc-xcc)**2 + (yc-ycc)**2)**0.5)
            if d < mind:
                mind = d
                iout = ic
        if iout > -1: # if we found a mega-yolo-found caption, add it
            fig_cap_pair_true.append([ibb,iout])
        else: # if no, mark as no caption
            fig_cap_pair_true.append([ibb,-1]) # nothing found

    # expand fig->caption for overlaps
    # NOTE: this associates each figure with a caption -- i.e., no "floating" captions
    # floating figures are OK however
    for f in fig_cap_pair_true:
        bf = boxes_true_fig[f[0]].copy() # fig box
        if f[1] != -1: # have a associated caption
            bc = boxes_true_cap[f[1]].copy() # cap box
            boxesOutTrue.append(bc)     
            # which side is caption on? 
            xcf,ycf = 0.5*(bf[0]+bf[2]), 0.5*(bf[1]+bf[3])
            xcc,ycc = 0.5*(bc[0]+bc[2]), 0.5*(bc[1]+bc[3])
            xShift = (xcc-xcf)/config.IMAGE_W; yShift = -1.0*(ycc-ycf)/config.IMAGE_H
            theta = np.arctan2(yShift,xShift)*180/np.pi
            # arctan2 in range [-180,180]
            if (theta < -70) and (theta > -110): # caption on bottom
                bf[0] = min(bc[0],bf[0]); bf[2] = max(bf[2],bc[2])
            elif (theta < 20) and (theta > -20): # caption on right
                bf[1]=min(bc[1],bf[1]); bf[3]=max(bf[3],bc[3])
            elif (theta > 160) or (theta < -160):
                bf[1]=min(bc[1],bf[1]); bf[3]=max(bf[3],bc[3])
        boxesOutTrue.append(bf) # either way, add it on
    # tack on others as well
    for tbox in true_others:
        boxesOutTrue.append(tbox.copy())   

    truebox1 = boxesOutTrue.copy() # replace
    return truebox1




def expand_found_area_above_cap(boxes_sq, labels_sq, scores_sq, bbsq, 
                               rotatedImage, LABELS, dfMS,
                               width=None, height=None):
    if dfMS is not None:
        fracx = dfMS['w'].values[0]*1.0/config.IMAGE_W
        fracy = dfMS['h'].values[0]*1.0/config.IMAGE_H 
    elif (width is not None) and (height is not None):
        fracx = width*1.0/config.IMAGE_W
        fracy = height*1.0/config.IMAGE_H 
    # now, do for found boxes
    # Note: in theory, you can do this in the step above, but here I'm just doing them together 
    # for clarity (true vs found)
    boxes_fig = []; boxes_cap = []; boxes_others = []
    labels_fig = []; labels_cap = []; labels_other = []
    scores_fig = []; scores_cap = []; scores_other = []
    fig_cap_pair = []
    boxesOut = []; labelsOut = []; scoresOut = []
    for box,l,s in zip(boxes_sq, labels_sq, scores_sq):
        if LABELS[int(l)] == 'figure caption':
            boxes_cap.append(box)
            labels_cap.append(l); scores_cap.append(s)
        elif LABELS[int(l)] == 'figure':
            boxes_fig.append(box)
            labels_fig.append(l); scores_fig.append(s)
        else:
            boxes_others.append(box)
            labels_other.append(l); scores_other.append(s)

    # pair
    for ibb,bb in enumerate(boxes_fig):
        mind = 5e15; iout = -1
        xc,yc = 0.5*(bb[0]+bb[2]),bb[3]
        if rotatedImage:
            xc, yc = bb[2], 0.5*(bb[1]+bb[3])
        for ic,bc in enumerate(boxes_cap): # find closest to bottom, not inside a square
            xcc,ycc= 0.5*(bc[0]+bc[2]),0.5*(bc[1]+bc[3])
            d = np.sum(((xc-xcc)**2 + (yc-ycc)**2)**0.5)
            if d < mind:
                mind = d
                iout = ic
        if iout > -1: # if we found a mega-yolo-found caption, add it
            fig_cap_pair.append([ibb,iout])
        else: # if no, mark as no caption
            fig_cap_pair.append([ibb,-1]) # nothing found

    # expand fig->caption for overlaps
    # NOTE: this associates each figure with a caption -- i.e., no "floating" captions
    # floating figures are OK however
    for f in fig_cap_pair:
        bf = np.array(boxes_fig[f[0]]) # fig box
        if f[1] != -1: # have a associated caption
            bc = boxes_cap[f[1]].copy() # cap box
            boxesOut.append(bc) # either way, add it on
            labelsOut.append(labels_cap[f[1]])
            scoresOut.append(scores_cap[f[1]])
            # which side is caption on?
            xcf,ycf = 0.5*(bf[0]+bf[2]), 0.5*(bf[1]+bf[3])
            xcc,ycc = 0.5*(bc[0]+bc[2]), 0.5*(bc[1]+bc[3])
            xShift = (xcc-xcf)/config.IMAGE_W; yShift = -1.0*(ycc-ycf)/config.IMAGE_H
            #xShift = (xcf-xcc)/IMAGE_W; yShift = (ycf-ycc)/IMAGE_H
            theta = np.arctan2(yShift,xShift)*180/np.pi
            # arctan2 in range [-180,180]
            if (theta < -70) and (theta > -110): # caption on bottom
                bf[0] = min(bc[0],bf[0]); bf[2] = max(bf[2],bc[2])
            elif (theta < 20) and (theta > -20): # caption on right
                bf[1]=min(bc[1],bf[1]); bf[3]=max(bf[3],bc[3])
            elif (theta > 160) or (theta < -160):
                bf[1]=min(bc[1],bf[1]); bf[3]=max(bf[3],bc[3])
        boxesOut.append(bf) # either way, add it on
        labelsOut.append(labels_fig[f[0]])
        scoresOut.append(scores_fig[f[0]])
    # for others
    for b,l,s in zip(boxes_others, labels_other, scores_other):
        boxesOut.append(b); labelsOut.append(l); 
        scoresOut.append(s)

    # yet again we have to do this... can we do this just once here?
    for ibox in range(len(boxesOut)):
        x1min,y1min,x1max,y1max = boxesOut[ibox]
        if labelsOut[ibox] == LABELS.index('figure'):
            for ibb,bb in enumerate(bbsq): # only do the once
                x2min, y2min, x2max, y2max = bb[0]/fracx, bb[1]/fracy, bb[2]/fracx, bb[3]/fracy
                isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                   (x2min,y2min,x2max,y2max))
                if isOverlapping:
                    boxesOut[ibox] = (min(x1min,x2min),
                                      min(y1min,y2min), max(x1max,x2max), max(y1max,y2max))       

    boxes_sq = boxesOut.copy(); labels_sq = labelsOut.copy(); scores_sq = scoresOut.copy()
    return boxes_sq, labels_sq, scores_sq

