# redo features only -- allow for other lists of features
import config

# # this supercedes what is in the config file
# feature_list = ['grayscale','fontsize','carea boxes','paragraph boxes','fraction of numbers in a word','fraction of letters in a word',
#                 'punctuation','x_ascenders','x_decenders','text angles', 'word confidences','Spacy POS','Spacy TAGs','Spacy DEPs']

feature_list = ['grayscale']
# call these something new?
binaries_file = 'model1_inverted_palletized'
#mode = 'P' # "L" is default for grayscale formatting
mode = 'L' # "L" is default for grayscale formatting

feature_list = ['grayscale','fontsize']
# call these something new?
binaries_file = 'model2'

feature_list = ['grayscale','fontsize','x_ascenders','x_decenders']
# call these something new?
binaries_file = 'model3'

feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences']
# call these something new?
binaries_file = 'model4'

feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
                'fraction of numbers in a word','fraction of letters in a word','punctuation']
# call these something new?
binaries_file = 'model5'
maxTag = 50 # trial? for fractin of ___ and punctuation

feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
                'fraction of numbers in a word','fraction of letters in a word','punctuation']
# call these something new?
binaries_file = 'model5_maxTag125'
maxTag = 125 # trial? for fractin of ___ and punctuation

feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
                'fraction of numbers in a word','fraction of letters in a word','punctuation', 
               'text angles']
# call these something new?
binaries_file = 'model6'

feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
                'fraction of numbers in a word','fraction of letters in a word','punctuation', 
               'text angles','Spacy POS']
# call these something new?
binaries_file = 'model7'

feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
                'fraction of numbers in a word','fraction of letters in a word','punctuation', 
               'text angles','Spacy POS','Spacy TAGs','Spacy DEPs']
# call these something new?
binaries_file = 'model8'

feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
                'fraction of numbers in a word','fraction of letters in a word','punctuation', 
               'text angles','Spacy POS','Spacy TAGs','Spacy DEPs']
# call these something new?
binaries_file = 'model8_pickle'

feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
                'fraction of numbers in a word','fraction of letters in a word','punctuation', 
               'text angles','Spacy POS','Spacy TAGs','Spacy DEPs']
# call these something new?
binaries_file = 'model8_noncom'

# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS','Spacy TAGs','Spacy DEPs']
# # call these something new?
# binaries_file = 'model8_noncomz'

# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS','Spacy TAGs','Spacy DEPs', 'paragraph boxes']
# # call these something new?
# binaries_file = 'model9'

# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS','Spacy TAGs','Spacy DEPs', 'paragraph boxes', 'carea boxes']
# # call these something new?
# binaries_file = 'model10'

# ----------------------------------------------

# easy parallel
import yt
yt.enable_parallelism()

from pathlib import Path
import time
from threading import Lock
from lxml import etree
import os
import shutil
import pandas as pd
import cv2 as cv
import numpy as np
#import xml.etree.ElementTree as ET
from glob import glob

from annotation_utils import get_all_ocr_files, make_ann_directories, collect_ocr_process_results, \
   get_makesense_info_and_years, get_years, get_cross_index, get_pdffigures_info, get_annotation_name, \
   true_box_caption_mod

from ocr_and_image_processing_utils import angles_results_from_ocr

from feature_generation_utils import generate_single_feature

from general_utils import parse_annotation

# general debug
debug = False

# ----------------------------------------------

# let's get all of the ocr files
ocrFiles = get_all_ocr_files()
# get important quantities from these files
if yt.is_root(): print('retreiving OCR data, this can take a moment...')
ws, paragraphs, squares, html, rotations,colorbars = collect_ocr_process_results(ocrFiles)
# create dataframe
df = pd.DataFrame({'ws':ws, 'paragraphs':paragraphs, 'squares':squares, 
                   'hocr':html, 'rotation':rotations, 'colorbars':colorbars})#, 'pdfwords':pdfwords})
df = df.drop_duplicates(subset='ws')
df = df.set_index('ws')

binaries_file2 = config.save_binary_dir + 'binaries'
if len(binaries_file)>0: #add
    binaries_file = binaries_file2 + '_' + binaries_file + '/'
else:
    binaries_file = binaries_file2 + '/'

def create_stuff(lock):
    if os.path.isfile(config.save_binary_dir + 'done'): os.remove(config.save_binary_dir + 'done') # test to make sure we don't move on in parallel too soon
    if yt.is_root():
        # check these all exist, but don't over write the directories like for annotaitons
        if not os.path.exists(binaries_file):
            os.mkdir(binaries_file)  
        # done
        with open(config.save_binary_dir + 'done','w') as ffd:
            print('done!',file=ffd)
        print(config.save_binary_dir + 'done')

# in theory, this should stop the parallel stuff until the folder
#. has been created, but I'm not 100% sure on this one
my_lock = Lock()
create_stuff(my_lock)

# get annotations
imgDirAnn = config.save_binary_dir + config.ann_name + str(int(config.IMAGE_H)) + 'x' + str(int(config.IMAGE_W))  + '_ann/'
# get all annotations
annotations = glob(imgDirAnn+'*.xml')

# storage
my_storage = {}
wsInds = np.linspace(0,len(annotations)-1,len(annotations)).astype('int')
# debug
#wsInds = wsInds[:2]
mod_output = 100
                   
                   

# lets do this thing...
if yt.is_root(): print('Making new features...')

for sto, iw in yt.parallel_objects(wsInds, config.nProcs, storage=my_storage):
    if iw%mod_output == 0: print('On ' + str(iw) + ' of ' + str(len(annotations)))

    sto.result_id = iw
        
    img_resize=(config.IMAGE_H, config.IMAGE_W)
    
    fname = annotations[iw].split('/')[-1].split('.xml')[0]
    floc = binaries_file + fname + '.npz'
    
    dfsingle = df.loc[fname+'.jpeg']
        
    # if we've made it this far, let's generate features
    feature_name = generate_single_feature(dfsingle, feature_list = feature_list, 
                                           binary_dir = binaries_file, 
                                           mode=mode, maxTag=maxTag)
    
    #import sys; sys.exit()
