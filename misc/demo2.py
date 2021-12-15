# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from sys import path
path.append('./demo')

from predictor import VisualizationDemo

import yt
yt.enable_parallelism()

from PIL import Image

nProcs = 6
# where to save images?
viz_output_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/MetricsResults/fbdetect/'

args_config_file = 'configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml'
args_opts = ['MODEL.WEIGHTS','/Users/jillnaiman/Downloads/model_final_trimmed.pth','MODEL.DEVICE','cpu']
#args_opts = '~/Downloads/model_final_trimmed.pth'

args_confidence_threshold = 0.5 # is this even used? ...

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args_config_file)
    cfg.merge_from_list(args_opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args_confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args_confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args_confidence_threshold
    cfg.freeze()
    return cfg


from detectron2.data import MetadataCatalog
MetadataCatalog.get("dla_val").thing_classes = ['text', 'title', 'list', 'table', 'figure']
mp.set_start_method("spawn", force=True)
#args = get_parser().parse_args()
#setup_logger(name="fvcore")
#logger = setup_logger()
#logger.info("Arguments: " + str(args))

cfg = setup_cfg()

demo = VisualizationDemo(cfg)


# read in lists
binary_dirs = 'binaries_model6_tfrecordz/'
use_valid = False # use test dataset for comparison
adder = ''
if use_valid: adder = '_valid'
import pickle
# build up filename
metrics_path = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/MetricsResults/'

pp = metrics_path
pp += binary_dirs.split('/')[0]
pp += adder
pp += '.pickle'
with open(pp, 'rb') as ff:
    icombo,imgs_name, truebox, pdfboxes, pdfrawboxes, captionText_figcap,\
                 bbox_figcap_pars,\
                 sboxes_cleaned, slabels_cleaned, sscores_cleaned,\
                 boxes_pdf, labels_pdf, scores_pdf,\
                 boxes_heur, labels_heur, scores_heur,\
                 boxes_heur2, labels_heur2, scores_heur2,\
                 boxes_par_found, labels_par_found, scores_par_found,\
                 boxes_sq1, labels_sq1, scores_sq1,\
                 boxes_sq2, labels_sq2, scores_sq2,\
                 boxes_sq3, labels_sq3, scores_sq3,\
                 boxes_sq4, labels_sq4, scores_sq4,\
                 boxes_sq5, labels_sq5, scores_sq5,\
                 truebox1,truebox2,truebox3,rotatedImage,LABELS,boxes1, scores1, labels1 = pickle.load(ff)


#img_path = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Pages/RandomSingleFromPDFIndexed/1993ApJ___403__202T_p1.jpeg'
img_paths = []
jpegPath = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Pages/RandomSingleFromPDFIndexed/'
for im in imgs_name:
    img_paths.append(jpegPath + im.split('/')[-1].split('.npz')[0]+'.jpeg')

# debug:
#img_paths = img_paths[:6]

wsInds = np.arange(0,len(img_paths))

import sys

my_storage = {}
for sto, iimg in yt.parallel_objects(wsInds, nProcs, storage=my_storage):
    sto.result_id = iimg
    img_path = img_paths[iimg]

    img = read_image(img_path, format="BGR")

    predictions, visualized_output = demo.run_on_image(img)

    # save
    out_filename = os.path.join(viz_output_dir, os.path.basename(img_path))
    visualized_output.save(out_filename)


    if len(predictions['instances']) > 0:
        _,height,width = predictions['instances'][0].pred_masks.numpy().shape
    else:
        ii = np.array(Image.open(img_path))
        height,width = ii.shape[0], ii.shape[1]
        del ii
    
    boxes = []; scores = []; classes = [];
    for ib in range(len(predictions['instances'])):
        boxes.append(predictions['instances'][ib].pred_boxes.tensor.numpy())
        scores.append(predictions['instances'][ib].scores.numpy())
        classes.append(predictions['instances'][ib].pred_classes.numpy())

    #all_boxes.append(boxes)

    if iimg%100 == 0: print('on', iimg, 'of', len(img_paths), flush=True)

    sto.result = [img_path, boxes, classes, scores,height,width]


if yt.is_root():
    img_names = []; boxes = []; classes=[]; scores = []; height = []; width=[]
    
    for ns,vals in sorted(my_storage.items()):
        if vals is not None:
            img_names.append(vals[0])
            boxes.append(vals[1])
            classes.append(vals[2])
            scores.append(vals[3])
            height.append(vals[4])
            width.append(vals[5])
            
    # build up filename
    pp = metrics_path
    pp += binary_dirs.split('/')[0]
    pp += adder
    pp += '_fbdetect'           
    pp += '.pickle'
    with open(pp, 'wb') as ff:
        pickle.dump([img_names,boxes,classes,scores,height,width], ff)
            
