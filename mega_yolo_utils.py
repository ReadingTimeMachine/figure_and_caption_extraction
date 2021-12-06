# stores ongoing stuff for mega-yolo annotation
import config

import numpy as np
import xml.etree.ElementTree as ET
import math
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
import glob
import pickle
from general_utils import parse_annotation
import imgaug as ia
from imgaug import augmenters as iaa

# some parameters for different architectures of YOLO
width_vec = [0.50, 0.75, 1.0, 1.25]
depth_vec = [0.33, 0.67, 1.0, 1.33]
versions = ['s', 'm', 'l', 'x']
threshold = 0.3
max_boxes = 150
batch_size = 32
image_size = 512
num_epochs = 300

import os

from general_utils import isRectangleOverlap, iou_orig

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
    t = glob.glob(classDir_main_to_imgs + '*')[0]

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
    depth = depth_vec[versions.index(version)]
    width = width_vec[versions.index(version)]

    # JPN add pseudo coloring
    if use_ps:
        inputs = tf.keras.layers.Input((image_size, image_size, n_features), dtype='float32')
        bn = tf.keras.layers.BatchNormalization(input_shape=[image_size, image_size, n_features], name='pseudoIn')(inputs)
        # add pseudo coloring
        pseudo_conv1 = keras.layers.Conv2D(filters=50, kernel_size=(1,1), strides=1, padding="SAME", activation="relu", 
                                input_shape=[None, None, 1], name='pseudo1')(bn) # into 20 psedo color channels
        pseudo_conv2 = keras.layers.Conv2D(filters=3, kernel_size=(1,1), strides=1, padding="SAME", activation="relu", 
                               name='pseudo2')(pseudo_conv1) # down to RGB channels
        # now back to our regularly scheduled program
        x = tf.nn.space_to_depth(pseudo_conv2, 2)
    else:
        #inputs = layers.Input([image_size, image_size, 3])
        inputs = layers.Input([image_size, image_size, n_features])
        x = tf.nn.space_to_depth(inputs, 2)    
    
    #inputs = layers.Input([image_size, image_size, n_features])
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
    y_true_1 = np.zeros((image_size // 32,
                            image_size // 32,
                            3, 5 + CLASS), np.float32)
    y_true_2 = np.zeros((image_size // 16,
                            image_size // 16,
                            3, 5 + CLASS), np.float32)
    y_true_3 = np.zeros((image_size // 8,
                            image_size // 8,
                            3, 5 + CLASS), np.float32)
    
    
    # if empty boxes, don't bother!
    if not tf.equal(tf.size(boxes),0):
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        #anchors = anchors
        box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
        box_size = boxes[:, 2:4] - boxes[:, 0:2]


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
            if type(c) != np.ndarray:
                c = labels[i].numpy().astype('int')


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
    ratio = tf.cast(tf.constant([image_size, image_size]) / grid_size, tf.float32)
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
def build_predict(weightsFile, anchorsFile, classDir_main_to_imgs, 
                  LABELS,version='l', debug=False, use_ps = False, 
                 use_tfrecords = True, n_features=None):
    # read in anchors
    with open(anchorsFile, 'rb') as f:
        anchors = pickle.load(f) 
        anchors = anchors.astype('float32')
    # get number of features
    if not use_tfrecords:
        n_features = get_n_features(classDir_main_to_imgs)
    if debug:
        print('n features=', n_features)
    
    model_predict = build_model(n_features, anchors, version, len(LABELS),training=False, use_ps=False)
    if debug:
        tf.keras.utils.plot_model(model_predict, "yolo_v5.png", show_shapes=True, 
                                  show_layer_names=True, expand_nested=False)
    model_predict.load_weights(weightsFile) # note there was a True) here and i took it out and now things work?  for REASONS.
    
    return model_predict


# ################################################
# # others
# def iou_orig(x1, y1, w1, h1, x2, y2, w2, h2, return_individual = False): 
#     '''
#     Calculate IOU between box1 and box2

#     Parameters
#     ----------
#     - x, y : box ***center*** coords
#     - w : box width
#     - h : box height
#     - return_individual: return intersection, union and IOU? default is False
    
#     Returns
#     -------
#     - IOU
#     '''   
#     xmin1 = x1 - 0.5*w1
#     xmax1 = x1 + 0.5*w1
#     ymin1 = y1 - 0.5*h1
#     ymax1 = y1 + 0.5*h1
#     xmin2 = x2 - 0.5*w2
#     xmax2 = x2 + 0.5*w2
#     ymin2 = y2 - 0.5*h2
#     ymax2 = y2 + 0.5*h2
#     interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
#     intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
#     inter = interx * intery
#     union = w1*h1 + w2*h2 - inter
#     iou = inter / (union + 1e-6)
#     if not return_individual:
#         return iou
#     else:
#         return inter, union, iou

# # edited SPACY glossary
# GLOSSARY = {
#     # POS tags
#     # Universal POS Tags
#     # http://universaldependencies.org/u/pos/
#     "ADJ": "adjective",
#     "ADP": "adposition",
#     "ADV": "adverb",
#     "AUX": "auxiliary",
#     "CONJ": "conjunction",
#     "CCONJ": "coordinating conjunction",
#     "DET": "determiner",
#     "INTJ": "interjection",
#     "NOUN": "noun",
#     "NUM": "numeral",
#     "PART": "particle",
#     "PRON": "pronoun",
#     "PROPN": "proper noun",
#     "PUNCT": "punctuation",
#     "SCONJ": "subordinating conjunction",
#     "SYM": "symbol",
#     "VERB": "verb",
#     "X": "other",
#     "EOL": "end of line",
#     "SPACE": "space",
#     # POS tags (English)
#     # OntoNotes 5 / Penn Treebank
#     # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
#     ".": "punctuation mark, sentence closer",
#     ",": "punctuation mark, comma",
#     "-LRB-": "left round bracket",
#     "-RRB-": "right round bracket",
#     "``": "opening quotation mark",
#     '""': "closing quotation mark",
#     "''": "closing quotation mark",
#     ":": "punctuation mark, colon or ellipsis",
#     "$": "symbol, currency",
#     "#": "symbol, number sign",
#     "AFX": "affix",
#     "CC": "conjunction, coordinating",
#     "CD": "cardinal number",
#     "DT": "determiner",
#     "EX": "existential there",
#     "FW": "foreign word",
#     "HYPH": "punctuation mark, hyphen",
#     "IN": "conjunction, subordinating or preposition",
#     "JJ": "adjective (English), other noun-modifier (Chinese)",
#     "JJR": "adjective, comparative",
#     "JJS": "adjective, superlative",
#     "LS": "list item marker",
#     "MD": "verb, modal auxiliary",
#     "NIL": "missing tag",
#     "NN": "noun, singular or mass",
#     "NNP": "noun, proper singular",
#     "NNPS": "noun, proper plural",
#     "NNS": "noun, plural",
#     "PDT": "predeterminer",
#     "POS": "possessive ending",
#     "PRP": "pronoun, personal",
#     "PRP$": "pronoun, possessive",
#     "RB": "adverb",
#     "RBR": "adverb, comparative",
#     "RBS": "adverb, superlative",
#     "RP": "adverb, particle",
#     "TO": 'infinitival "to"',
#     "UH": "interjection",
#     "VB": "verb, base form",
#     "VBD": "verb, past tense",
#     "VBG": "verb, gerund or present participle",
#     "VBN": "verb, past participle",
#     "VBP": "verb, non-3rd person singular present",
#     "VBZ": "verb, 3rd person singular present",
#     "WDT": "wh-determiner",
#     "WP": "wh-pronoun, personal",
#     "WP$": "wh-pronoun, possessive",
#     "WRB": "wh-adverb",
#     "SP": "space (English), sentence-final particle (Chinese)",
#     "ADD": "email",
#     "NFP": "superfluous punctuation",
#     "GW": "additional word in multi-word expression",
#     "XX": "unknown",
#     "BES": 'auxiliary "be"',
#     "HVS": 'forms of "have"',
#     # Dependency Labels (English)
#     # ClearNLP / Universal Dependencies
#     # https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
#     "acl": "clausal modifier of noun (adjectival clause)",
#     "acomp": "adjectival complement",
#     "advcl": "adverbial clause modifier",
#     "advmod": "adverbial modifier",
#     "agent": "agent",
#     "amod": "adjectival modifier",
#     "appos": "appositional modifier",
#     "attr": "attribute",
#     "aux": "auxiliary",
#     "auxpass": "auxiliary (passive)",
#     "case": "case marking",
#     "cc": "coordinating conjunction",
#     "ccomp": "clausal complement",
#     "clf": "classifier",
#     "complm": "complementizer",
#     "compound": "compound",
#     "conj": "conjunct",
#     "cop": "copula",
#     "csubj": "clausal subject",
#     "csubjpass": "clausal subject (passive)",
#     "dative": "dative",
#     "dep": "unclassified dependent",
#     "det": "determiner",
#     "discourse": "discourse element",
#     "dislocated": "dislocated elements",
#     "dobj": "direct object",
#     "expl": "expletive",
#     "fixed": "fixed multiword expression",
#     "flat": "flat multiword expression",
#     "goeswith": "goes with",
#     "hmod": "modifier in hyphenation",
#     "hyph": "hyphen",
#     "infmod": "infinitival modifier",
#     "intj": "interjection",
#     "iobj": "indirect object",
#     "list": "list",
#     "mark": "marker",
#     "meta": "meta modifier",
#     "neg": "negation modifier",
#     "nmod": "modifier of nominal",
#     "nn": "noun compound modifier",
#     "npadvmod": "noun phrase as adverbial modifier",
#     "nsubj": "nominal subject",
#     "nsubjpass": "nominal subject (passive)",
#     "nounmod": "modifier of nominal",
#     "npmod": "noun phrase as adverbial modifier",
#     "num": "number modifier",
#     "number": "number compound modifier",
#     "nummod": "numeric modifier",
#     "oprd": "object predicate",
#     "obj": "object",
#     "obl": "oblique nominal",
#     "orphan": "orphan",
#     "parataxis": "parataxis",
#     "partmod": "participal modifier",
#     "pcomp": "complement of preposition",
#     "pobj": "object of preposition",
#     "poss": "possession modifier",
#     "possessive": "possessive modifier",
#     "preconj": "pre-correlative conjunction",
#     "prep": "prepositional modifier",
#     "prt": "particle",
#     "punct": "punctuation",
#     "quantmod": "modifier of quantifier",
#     "rcmod": "relative clause modifier",
#     "relcl": "relative clause modifier",
#     "reparandum": "overridden disfluency",
#     "root": "root",
#     "vocative": "vocative",
#     "xcomp": "open clausal complement",
#     # Named Entity Recognition
#     # OntoNotes 5
#     # https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf
#     "PERSON": "People, including fictional",
#     "NORP": "Nationalities or religious or political groups",
#     "FACILITY": "Buildings, airports, highways, bridges, etc.",
#     "FAC": "Buildings, airports, highways, bridges, etc.",
#     "ORG": "Companies, agencies, institutions, etc.",
#     "GPE": "Countries, cities, states",
#     "LOC": "Non-GPE locations, mountain ranges, bodies of water",
#     "PRODUCT": "Objects, vehicles, foods, etc. (not services)",
#     "EVENT": "Named hurricanes, battles, wars, sports events, etc.",
#     "WORK_OF_ART": "Titles of books, songs, etc.",
#     "LAW": "Named documents made into laws.",
#     "LANGUAGE": "Any named language",
#     "DATE": "Absolute or relative dates or periods",
#     "TIME": "Times smaller than a day",
#     "PERCENT": 'Percentage, including "%"',
#     "MONEY": "Monetary values, including unit",
#     "QUANTITY": "Measurements, as of weight or distance",
#     "ORDINAL": '"first", "second", etc.',
#     "CARDINAL": "Numerals that do not fall under another type",
#     # Named Entity Recognition
#     # Wikipedia
#     # http://www.sciencedirect.com/science/article/pii/S0004370212000276
#     # https://pdfs.semanticscholar.org/5744/578cc243d92287f47448870bb426c66cc941.pdf
#     "PER": "Named person or family.",
#     "MISC": "Miscellaneous entities, e.g. events, nationalities, products or works of art",
#     # https://github.com/ltgoslo/norne
#     "EVT": "Festivals, cultural events, sports events, weather phenomena, wars, etc.",
#     "PROD": "Product, i.e. artificially produced entities including speeches, radio shows, programming languages, contracts, laws and ideas",
#     "DRV": "Words (and phrases?) that are dervied from a name, but not a name in themselves, e.g. 'Oslo-mannen' ('the man from Oslo')",
#     "GPE_LOC": "Geo-political entity, with a locative sense, e.g. 'John lives in Spain'",
#     "GPE_ORG": "Geo-political entity, with an organisation sense, e.g. 'Spain declined to meet with Belgium'",
# }

# # Also, the individual lists of things
# # this is from: https://stackoverflow.com/questions/58215855/how-to-get-full-list-of-pos-tag-and-dep-in-spacy
# TAG_LIST = np.unique(np.append(['$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC',
#         'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
#         'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP',
#         'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD',
#         'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP',
#         '``'],[":",".",",","-LRB-","-RRB-","``","\"\"","''",",","$","#","AFX","CC","CD","DT","EX","FW","HYPH","IN","JJ","JJR","JJS","LS","MD","NIL","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB","ADD","NFP","GW","XX","BES","HVS","_SP"])).tolist()
# POS_LIST = np.unique(np.append(['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
#         'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SPACE', 'SYM', 'VERB',
#         'X'],["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"])).tolist()
# DEP_LIST = np.unique(np.append(['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod',
#         'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp',
#         'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det',
#         'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod',
#         'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp',
#         'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct',
#         'quantmod', 'relcl', 'xcomp'],["ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl", "intj", "mark", "meta", "neg", "nn", "npmod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp", "nummod"])).tolist()


# # from other metrics: https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/src/evaluators/pascal_voc_evaluator.py
# def calculate_ap_every_point(rec, prec):
#     mrec = []
#     mrec.append(0)
#     [mrec.append(e) for e in rec]
#     mrec.append(1)
#     mpre = []
#     mpre.append(0)
#     [mpre.append(e) for e in prec]
#     mpre.append(0)
#     for i in range(len(mpre) - 1, 0, -1):
#         mpre[i - 1] = max(mpre[i - 1], mpre[i])
#     ii = []
#     for i in range(len(mrec) - 1):
#         if mrec[1:][i] != mrec[0:-1][i]:
#             ii.append(i + 1)
#     ap = 0
#     for i in ii:
#         ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
#     return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


# def calculate_ap_11_point_interp(rec, prec, recall_vals=11):
#     mrec = []
#     # mrec.append(0)
#     [mrec.append(e) for e in rec]
#     # mrec.append(1)
#     mpre = []
#     # mpre.append(0)
#     [mpre.append(e) for e in prec]
#     # mpre.append(0)
#     recallValues = np.linspace(0, 1, recall_vals)
#     recallValues = list(recallValues[::-1])
#     rhoInterp = []
#     recallValid = []
#     # For each recallValues (0, 0.1, 0.2, ... , 1)
#     for r in recallValues:
#         # Obtain all recall values higher or equal than r
#         argGreaterRecalls = np.argwhere(mrec[:] >= r)
#         pmax = 0
#         # If there are recalls above r
#         if argGreaterRecalls.size != 0:
#             pmax = max(mpre[argGreaterRecalls.min():])
#         recallValid.append(r)
#         rhoInterp.append(pmax)
#     # By definition AP = sum(max(precision whose recall is above r))/11
#     ap = sum(rhoInterp) / len(recallValues)
#     # Generating values for the plot
#     rvals = []
#     rvals.append(recallValid[0])
#     [rvals.append(e) for e in recallValid]
#     rvals.append(0)
#     pvals = []
#     pvals.append(0)
#     [pvals.append(e) for e in rhoInterp]
#     pvals.append(0)
#     # rhoInterp = rhoInterp[::-1]
#     cc = []
#     for i in range(len(rvals)):
#         p = (rvals[i], pvals[i - 1])
#         if p not in cc:
#             cc.append(p)
#         p = (rvals[i], pvals[i])
#         if p not in cc:
#             cc.append(p)
#     recallValues = [i[0] for i in cc]
#     rhoInterp = [i[1] for i in cc]
#     return [ap, rhoInterp, recallValues, None]

# # calculate overlaps -- https://www.tutorialspoint.com/rectangle-overlap-in-python
# # and https://stackoverflow.com/questions/40795709/checking-whether-two-rectangles-overlap-in-python-using-two-bottom-left-corners
# def isRectangleOverlap(R1, R2):
#     if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
#         return False
#     else:
#         return True

# FP/FN/TP calcs:
def calc_metrics(truebox1, boxes_sq, labels_sq, scores_sq, LABELS,ioumin,
                years=[], iioumin=-1, iscoremin=-1, 
                TPyear = [], FPyear=[], totalTrueyear=[], FNyear=[], year=[],
                totalTruev=[], TPv=[], FPv=[],FNv=[], return_pairs = False):
    """
    truebox1: trueboxes in YOLO coords (typically 512x512) (xmin,ymin,xmax,ymax, LABEL_INDEX+1)
    boxes_sq: found boxes in YOLO coords (xmin,ymin,xmax,ymax)
    labels_sq: found box labels (LABEL_INDEX) -- NOTE no +1!
    scores_sq: found box score (0.0-1.0)
    LABELS: list of labels (strings), e.g. ['figure', 'figure caption', 'table']
    """
    # checks
    if iioumin == -1 and iscoremin != -1: 
        print('not supported for different params for iioumin & iscoremin')
        import sys; sys.exit()
    # other checks
    if iioumin == -1 and iscoremin == -1:
        totalTruev = np.zeros(len(LABELS))
        FNv = np.zeros(len(LABELS))
        FPv = np.zeros(len(LABELS))
        TPv = np.zeros(len(LABELS))
        if len(years)>0:
            totalTrueyear = np.zeros([len(years),len(LABELS)])
            FNyear = np.zeros([len(years),len(LABELS)])
            FPyear = np.zeros([len(years),len(LABELS)])
            TPyear = np.zeros([len(years),len(LABELS)])

    # make array of total number of boxes true/found, tag extra found with a FP tag
    # loop and find closest boxes
    true_found_index = []; true_found_labels = []; trueCaps = []; foundCaps = []; #iouSave = []
    for it,tbox in enumerate(truebox1): # if there really is a box
        w2, h2 = tbox[2]-tbox[0], tbox[3]-tbox[1]
        x2, y2 = tbox[0]+0.5*w2, tbox[1]+0.5*h2
        # just for found captions
        if iioumin != -1:
            totalTruev[int(tbox[4]-1),iioumin,iscoremin] += 1
            if len(years)>0:
                totalTrueyear[years==year,int(tbox[4]-1),iioumin,iscoremin] += 1                
        else:
            totalTruev[int(tbox[4]-1)] += 1
            if len(years) >0:
                totalTrueyear[years==year,int(tbox[4]-1)] += 1                
            
        trueCaps.append(tbox)
        # find greatest IOU -- literally there is a better algorithm here to do this
        iouMax = -10
        foundBox = False
        indFound = [it,-1]; labelsFound = [tbox[-1]-1, -1]; foundCapHere = []
        for ib,b in enumerate(boxes_sq):
            isOverlapping = isRectangleOverlap(tbox[:-1],b)
            w1, h1 = b[2]-b[0], b[3]-b[1]
            x1, y1 = b[0]+0.5*w1, b[1]+0.5*h1
            iou1 = iou_orig(x1,y1,w1,h1, x2,y2,w2,h2)
            # a win!
            if (iou1 > iouMax) and (iou1 > ioumin) and isOverlapping: # check for overlap
                iouMax = iou1
                indFound[-1] = ib
                labelsFound[-1] = labels_sq[ib]
                foundCapHere = b
                #print('yes')
        true_found_index.append(indFound); true_found_labels.append(labelsFound); 
        if len(foundCapHere) > 0: foundCaps.append(foundCapHere)
        
    # count
    # save pairs, unfound trues, miss-found founds
    true_found_pairs = []
    for ti,tl in zip(true_found_index, true_found_labels): # ti = [true index, found index]
        ind = int(tl[0]) # index is true's label
        if ti[-1] == -1: # didn't find anything
            if iioumin != -1:
                FNv[ind,iioumin,iscoremin] +=1
                if len(years)>0:
                    FNyear[years==year,ind,iioumin,iscoremin] += 1
            else:
                FNv[ind] += 1
                if len(years)>0:
                    FNyear[years==year,ind] +=1
            # save as a true w/o a found
            true_found_pairs.append( (truebox1[ti[0]], -1) )
        elif ti[-1] != -1 and tl[0] != tl[1]: # overlap of boxes, but wrong things -- count as FN for this true
            if iioumin != -1:
                FNv[ind,iioumin,iscoremin] +=1
                if len(years)>0:
                    FNyear[years==year,ind,iioumin,iscoremin] += 1
            else:
                FNv[ind] += 1
                if len(years)>0:
                    FNyear[years==year,ind] +=1
            # save as a true w/o a found
            true_found_pairs.append( (truebox1[ti[0]], -1) )
        elif ti[-1] != -1 and tl[0] == tl[1]: # found a box AND its the right one!
            if iioumin != -1:
                TPv[ind,iioumin,iscoremin] +=1
                if len(years)>0:
                    TPyear[years==year,ind,iioumin,iscoremin] += 1
            else:
                TPv[ind] += 1
                if len(years)>0:
                    TPyear[years==year,ind] +=1
            #try:
            true_found_pairs.append( (truebox1[ti[0]], (boxes_sq[ti[1]][0],boxes_sq[ti[1]][1],
                                                    boxes_sq[ti[1]][2],boxes_sq[ti[1]][3],labels_sq[ti[1]])) )
#             except:
#                 print('ti',ti)
#                 print(' ')
#                 print('truebox',truebox1)
#                 print(' ')
#                 print('boxes_sq', boxes_sq)
#                 print(' ')
#                 print('labels_sq', labels_sq)
#                 import sys; sys.exit()
            
    # do we have extra found boxes?
    if len(boxes_sq) > len(trueCaps):
        for ib, b in enumerate(boxes_sq):
            if len(true_found_index) > 0: # we have some trues
                if ib not in np.array(true_found_index)[:,1].tolist(): # but we don't have this particular found matched to a true
                    ind = labels_sq[ib] # label will be found label -- mark as a FP for this label
                    if iioumin != -1:
                        FPv[ind,iioumin,iscoremin] +=1
                        if len(years)>0:
                            FPyear[years==year,ind,iioumin,iscoremin] += 1
                    else:
                        FPv[ind] += 1
                        if len(years)>0:
                            FPyear[years==year,ind] +=1
                    # mark as a found w/o a true
                    true_found_pairs.append( (-1, (boxes_sq[ib][0],boxes_sq[ib][1],
                                                    boxes_sq[ib][2],boxes_sq[ib][3],labels_sq[ib])) )
            elif len(true_found_index) == 0: # there is nothing true, any founds are FP
                ind = labels_sq[ib]
                if iioumin != -1:
                    FPv[ind,iioumin,iscoremin] +=1
                    if len(years)>0:
                        FPyear[years==year,ind,iioumin,iscoremin] += 1
                else:
                    FPv[ind] += 1
                    if len(years)>0:
                        FPyear[years==year,ind] +=1
                # mark as a found w/o a true
                true_found_pairs.append( (-1, (boxes_sq[ib][0],boxes_sq[ib][1],
                                                boxes_sq[ib][2],boxes_sq[ib][3],labels_sq[ib])) )

               
    if len(years)>0:
        if not return_pairs:
            return totalTruev, TPv, FPv, FNv, totalTrueyear, TPyear, FPyear, FNyear
        else:
            return totalTruev, TPv, FPv, FNv, totalTrueyear, TPyear, FPyear, FNyear, true_found_pairs
    else:
        if not return_pairs:
            return totalTruev, TPv, FPv, FNv
        else:
            return totalTruev, TPv, FPv, FNv,true_found_pairs

    
# this does even splitting of unbalanced classes:
def train_test_valid_split(X, y, train_size = 0.75, valid_size = 0.15, test_size = 0.1, img_links = None, 
                           asInts = True, reNorm = True, textClassification = False, shuffle=True):
    X_train = np.array([]); y_train = np.array([])
    X_valid = np.array([]); y_valid = np.array([])
    X_test = np.array([]); y_test = np.array([])
    links_train = np.array([]); links_valid = np.array([]); links_test = np.array([])
    if img_links is not None: img_links = np.array(img_links)
    for ind in np.unique(y):
        # subset
        inds = np.where(y == ind)[0]
        nSub = len(inds)
        itrain = int(round(nSub*train_size))
        ivalid = int(round(nSub*valid_size))
        itest = nSub-itrain-ivalid

        # take out training set
        if len(X_train) == 0:
            if not textClassification:
                X_train = X[inds[0:itrain],:,:]
            else:
                X_train = X[inds[0:itrain]]
            if img_links is not None:
                links_train = img_links[inds[0:itrain]]
        else:
            if not textClassification:
                X_train = np.concatenate((X_train, X[inds[0:itrain],:,:]), axis=0)
            else:
                X_train = np.concatenate((X_train, X[inds[0:itrain]]), axis=0)
                
            if img_links is not None:
                links_train = np.concatenate((links_train, img_links[inds[0:itrain]]), axis=0)
        y_train = np.append(y_train,y[inds[0:itrain]] )

        # take out validataion set
        if len(X_valid) == 0:
            if not textClassification:
                X_valid = X[inds[itrain:itrain+ivalid],:,:]
            else:
                X_valid = X[inds[itrain:itrain+ivalid]]
            if img_links is not None:
                links_valid = img_links[inds[itrain:itrain+ivalid]]
        else:
            if not textClassification:
                X_valid = np.concatenate((X_valid, X[inds[itrain:itrain+ivalid],:,:]), axis=0)
            else:
                X_valid = np.concatenate((X_valid, X[inds[itrain:itrain+ivalid]]), axis=0)
                
            if img_links is not None:
                links_valid = np.concatenate((links_valid, img_links[inds[itrain:itrain+ivalid]]), axis=0)
        y_valid = np.append(y_valid,y[inds[itrain:itrain+ivalid]] )

        # take out test set
        if len(X_test) == 0:
            if not textClassification:            
                X_test = X[inds[itrain+ivalid:],:,:]
            else:
                X_test = X[inds[itrain+ivalid:]]
            if img_links is not None:
                links_test = img_links[inds[itrain+ivalid:]]
        else:
            if not textClassification:            
                X_test = np.concatenate((X_test, X[inds[itrain+ivalid:],:,:]), axis=0)
            else:
                X_test = np.concatenate((X_test, X[inds[itrain+ivalid:]]), axis=0)
                    
            if img_links is not None:
                links_test = np.concatenate((links_test, img_links[inds[itrain+ivalid:]]), axis=0)
        y_test = np.append(y_test,y[inds[itrain+ivalid:]] )

    # lastly - we gotta shuffle!
    ind_train = np.random.choice(range(len(y_train)),len(y_train),replace=False)
    if not textClassification:            
        X_train = X_train[ind_train, :, :]
    else:
        X_train = X_train[ind_train]
        
    y_train = y_train[ind_train]

    ind_valid = np.random.choice(range(len(y_valid)),len(y_valid),replace=False)
    if not textClassification:            
        X_valid = X_valid[ind_valid, :, :]
    else:
        X_valid = X_valid[ind_valid]
        
    y_valid = y_valid[ind_valid]

    ind_test = np.random.choice(range(len(y_test)),len(y_test),replace=False)
    #print(ind_test)
    if not textClassification:            
        X_test = X_test[ind_test, :, :]
    else:
        X_test = X_test[ind_test]
        
    y_test = y_test[ind_test]
    
    # make sure outputs are incorrect format
    if asInts and not textClassification:
        y_train = y_train.astype('uint8')
        y_valid = y_valid.astype('uint8')
        y_test = y_test.astype('uint8')
    if textClassification:
        y_train = y_train.astype('int32')
        y_valid = y_valid.astype('int32')
        y_test = y_test.astype('int32')
        
        
    # rescale things like in examples?
    if reNorm and not textClassification:
        X_mean = X_train.mean(axis=0, keepdims=True)
        X_std = X_train.std(axis=0, keepdims=True) + 1e-7 # the last bit here is to avoid a division by 0
        X_train = (X_train - X_mean) / X_std
        X_valid = (X_valid - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

    if shuffle: # randomly shuffle indicies
        #inds = np.random.choice(range(len(y_train)-1),len(y_train),replace=False) # train
        inds = np.arange(len(y_train))
        np.random.shuffle(inds)
        X_train = X_train[inds,...]
        y_train = y_train[inds,...]
        if img_links is not None:
            links_train = links_train[inds,...]
        #inds = np.random.choice(range(len(y_valid)-1),len(y_valid),replace=False) # valid
        inds = np.arange(len(y_valid))
        np.random.shuffle(inds)
        X_valid = X_valid[inds,...]
        y_valid = y_valid[inds,...]
        if img_links is not None:
            links_valid = links_valid[inds,...]
        #inds = np.random.choice(range(len(y_test)-1),len(y_test),replace=False) # valid
        inds = np.arange(len(y_test))
        np.random.shuffle(inds)
        X_test = X_test[inds,...]
        y_test = y_test[inds,...]
        if img_links is not None:
            links_test = links_test[inds,...]
        del inds

    if img_links is None:
        return [X_train, y_train, X_valid, y_valid, X_test, y_test]
    else: # gotta return links to images
        return [X_train, y_train, X_valid, y_valid, X_test, y_test, links_train, links_valid, links_test]
        
        
        
########## DATA PREP #####################

def csv_gen(split, splitsDir = None):
    if splitsDir is None: splitsDir = config.tmp_storage_dir
    ftrain = open(splitsDir+'train.csv','r')
    fvalid = open(splitsDir+'valid.csv','r')
    ftest = open(splitsDir+'test.csv','r')    
    while True:
    #for i in range(100000):
        if b'train' in bytes(split, encoding='utf8'): # NOTE, must be bytes here!!!
            line = ftrain.readline()
            if line == "": # if EOF => loop back to start
                ftrain.seek(0)
                line = ftrain.readline()
        elif b'valid' in bytes(split, encoding='utf8'):
            line = fvalid.readline()
            if line == "": # if EOF => loop back to start
                fvalid.seek(0)
                line = fvalid.readline()
        elif b'test' in bytes(split, encoding='utf8'):
            line = ftest.readline()
            if line == "": # if EOF => loop back to start
                ftest.seek(0)
                line = ftest.readline()
                # make sure if we are evaluating with the test set 
                #  we don't loop back to the start of the file
                break 
        else:
            print('NOPE!')
            sys.exit()
        yield line
        


# def dataset_gen(split, dataset_generator_csv, batch_size, feature_dir,annotation_dir):
#     while True:
#         true_boxes = []; imgs = []; files = []
        
#         while len(files) < batch_size:
#             line = next(dataset_generator_csv)
#             # if type(split) == str:
#             #     if b
#             #     # if b'train' in bytes(split, encoding='utf8'): # NOTE, must be bytes here!!!
#             #     #     line = next(train_gen_csv)
#             #     # elif b'valid' in bytes(split, encoding='utf8'):
#             #     #     line = next(valid_gen_csv)
#             #     # elif b'test' in bytes(split, encoding='utf8'):
#             #     #     line = next(test_gen_csv)
#             # else:
#             #     if b'train' in split: # NOTE, must be bytes here!!!
#             #         line = next(train_gen_csv)
#             #     elif b'valid' in split:
#             #         line = next(valid_gen_csv)
#             #     elif b'test' in split:
#             #         line = next(test_gen_csv)
            
#             files.append(line.strip())

#         # parse and get full names
#         try:
#             imgs_name, bbox = parse_annotation(files, LABELS, 
#                                                feature_dir=feature_dir,
#                                                annotation_dir=annotation_dir)
#         except:
#             print('error parsing:', imgs_name)
#         # do a debug check
#         for im in imgs_name:
#             if '.npz' not in im:
#                 print('no np file')
#                 import sys; sys.exit()
        
#         # read in and keep images -- npy files
#         for im in imgs_name:
#             b = np.load(im)['arr_0']
            
#             ########### DEBUGGING ##########
#             #b = b[:,:,:3]
#             ################################
            
#             # convert 0-1
#             b = b/255.0
#             imgs.append(b)
        
#         # finally, format for output
#         y_true1, y_true2, y_true3 = [],[],[]
#         for b in bbox:
#             y1,y2,y3= process_box(b[:,:4], b[:,4].astype('int'),anchors,CLASS)
#             y_true1.append(y1); y_true2.append(y2); y_true3.append(y3)
        
#         img = tf.cast(np.array(imgs), tf.float32)        
#         yield img, tf.cast(y_true1, tf.float32), tf.cast(y_true2, tf.float32), tf.cast(y_true3, tf.float32)
        

# def dataset_gen_for_aug(split, batch_size, gen_in, 
#                         feature_dir, annotation_dir): # for training/validation datasets
#     while True:
#         true_boxes = []; imgs = []; files = []
        
#         while len(files) < batch_size:
#             line = next(gen_in)
#             # if type(split) == str:
#             #     if b'train' in bytes(split, encoding='utf8'): # NOTE, must be bytes here!!!
#             #         line = next(train_gen_csv)
#             #     elif b'valid' in bytes(split, encoding='utf8'):
#             #         line = next(valid_gen_csv)
#             #     elif b'test' in bytes(split, encoding='utf8'):
#             #         line = next(test_gen_csv)
#             # else:
#             #     if b'train' in split: # NOTE, must be bytes here!!!
#             #         line = next(train_gen_csv)
#             #     elif b'valid' in split:
#             #         line = next(valid_gen_csv)
#             #     elif b'test' in split:
#             #         line = next(test_gen_csv)
            
#             files.append(line.strip())

#         # parse and get full names
#         imgs_name, bbox = parse_annotation(files, LABELS, 
#                                                feature_dir=feature_dir,
#                                                annotation_dir=annotation_dir)

#         # do a debug check
#         for im in imgs_name:
#             if '.npz' not in im:
#                 print('no np file')
#                 import sys; sys.exit()
        
#         # read in and keep images -- npy files
#         for im in imgs_name:
#             b = np.load(im)['arr_0']
#             # convert 0-1
#             b = b/255.0
#             imgs.append(b)
                
#         img = tf.cast(np.array(imgs), tf.float32)        
#         yield img, tf.cast(bbox, tf.float32)
        

# def get_dataset(split, gen_in, labels, batch_size, 
#                 annotation_dir='', feature_dir='', use_aug=True):
#     if use_aug and ('test' not in split.lower()):
#         dataset = tf.data.Dataset.from_generator(dataset_gen_for_aug, 
#                                                  args=[split, batch_size, gen_in, 
#                                                        annotation_dir, feature_dir],
#                                                  output_types = (tf.float32, tf.float32))
#     else:
#         dataset = tf.data.Dataset.from_generator(dataset_gen, 
#                                                  args=[split, batch_size, gen_in, 
#                                                       annotation_dir, feature_dir],
#                                              output_types = (tf.float32, tf.float32, 
#                                                              tf.float32, tf.float32))
                                             
#     dataset = dataset.prefetch(10)
    
#     # maybe?
#     iterator = iter(dataset)

#     #return dataset
#     return iterator


def augmentation_generator(yolo_dataset, anchors, CLASS, flipUpDown=False):
    '''
    Augmented batch generator from a yolo dataset

    Parameters
    ----------
    - YOLO dataset
    
    Returns
    -------
    - augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    '''
    for batch in yolo_dataset:
        # conversion tensor->numpy
        img = batch[0].numpy()
        boxes = batch[1].numpy()
        # conversion bbox numpy->ia object
        ia_boxes = []
        if len(boxes) > 0:
            for i in range(img.shape[0]):
                ia_bbs = [ia.BoundingBox(x1=bb[0],
                                           y1=bb[1],
                                           x2=bb[2],
                                           y2=bb[3]) for bb in boxes[i]
                          if (bb[0] + bb[1] +bb[2] + bb[3] > 0)]
                ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(config.IMAGE_W, config.IMAGE_H)))
        # data augmentation
        if flipUpDown:
            seq = iaa.Sequential([
              iaa.Fliplr(0.5),
              iaa.Flipud(0.5),
              iaa.Multiply((0.4, 1.6)), # change brightness
              iaa.Affine(rotate=[0,90,270])
              #iaa.ContrastNormalization((0.5, 1.5)),
              #iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
              ])
        else:
            seq = iaa.Sequential([
              iaa.Fliplr(0.5),
              iaa.Multiply((0.4, 1.6)), # change brightness
              #iaa.ContrastNormalization((0.5, 1.5)),
              #iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
              iaa.Affine(rotate=[0,90,270])
              ])

        #seq = iaa.Sequential([])
        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        # conversion ia object -> bbox numpy
        if len(boxes) > 0:
            for i in range(img.shape[0]):
                boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
                for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                    boxes[i,j,0] = bb.x1
                    boxes[i,j,1] = bb.y1
                    boxes[i,j,2] = bb.x2
                    boxes[i,j,3] = bb.y2
        # conversion numpy->tensor
        #batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
        #batch = (img_aug, boxes)
        #yield batch
                # finally, format for output
        y_true1, y_true2, y_true3 = [],[],[]
        for b in boxes:
            y1,y2,y3= process_box(b[:,:4], b[:,4].astype('int'),anchors,CLASS)
            y_true1.append(y1); y_true2.append(y2); y_true3.append(y3)
        # if there is no box, do something different
        if len(boxes) == 0:
            # fake a box
            b = np.array([[111.,  59., 403., 364. ,  4.]])
            y1,y2,y3= process_box(b[:,:4], b[:,4].astype('int'),anchors,CLASS)
            y1[:] = 0; y2[:]=0;y3[:]=0
            y_true1.append(y1); y_true2.append(y2); y_true3.append(y3)
       
        img = tf.cast(np.array(img_aug), tf.float32)    
        yield img, tf.cast(y_true1, tf.float32), tf.cast(y_true2, tf.float32), tf.cast(y_true3, tf.float32)
