from glob import glob
import xml.etree.ElementTree as ET
from scipy import stats
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
import pickle
import os

# stuff
import config
from general_utils import parse_annotation

depth_vec = config.depth_vec
versions = config.versions
width_vec = config.width_vec
image_size = config.IMAGE_H # assume square I guess?
threshold = config.threshold
max_boxes = config.max_boxes

def parse_annotations_to_labels(classDir_main_to, testListFile, benchmark=False):
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

    # NEXT: do a quick test run-through of the data generator for train/test splits
    X_full = np.array(annotations)
    Y_full_str = np.array([]) # have to loop and give best guesses for the pages that have multiple images/classes in them
    slabels = []
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
    
    return LABELS, labels, slabels, CLASS, annotations, Y_full


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
def build_predict(weightsFile, anchorsFile, classDir_main_to_imgs, 
                  LABELS,version='l', debug=False, use_ps = False):
    # read in anchors
    with open(anchorsFile, 'rb') as f:
        anchors = pickle.load(f) 
        anchors = anchors.astype('float32')
    # get number of features
    n_features = get_n_features(classDir_main_to_imgs)
    if debug:
        print('n features=', n_features)
    
    model_predict = build_model(n_features, anchors, version, len(LABELS),training=False, use_ps=False)
    if debug:
        tf.keras.utils.plot_model(model_predict, "yolo_v5.png", show_shapes=True, 
                                  show_layer_names=True, expand_nested=False)
    model_predict.load_weights(weightsFile) # note there was a True) here and i took it out and now things work?  for REASONS.
    
    return model_predict



############# LOAD ANNOTATIONS #####################
def get_true_boxes(a,LABELS, badskews, badannotations, annotation_dir='', feature_dir=''):
    # get annotations, use pdffigures2 boxes as well
    years_ind = []
    imgs_name, true_boxes, pdfboxes, \
       pdfrawboxes = parse_annotation([a], LABELS,feature_dir = feature_dir,
                     annotation_dir = annotation_dir, parse_pdf=True)    

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
    iiname = imgs_name[0]
    iiname = iiname[:iiname.rfind('.')]
    #print(iiname)
    if (iiname.split('/')[-1] in badskews) or (iiname.split('/')[-1] in badannotations):
        print('bad skew or annotation for', icombo)
        ##continue
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
    
    return imgs_name, pdfboxes, pdfrawboxes,years_ind