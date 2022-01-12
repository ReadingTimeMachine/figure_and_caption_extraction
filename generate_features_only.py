# redo features only -- allow for other lists of features
import config
# for tfrecords -- set to None for re-do of splits
splits_directory = config.tmp_storage_dir
mode = 'L' # "L" is default for grayscale formatting
maxTag = 125 # trial? for fraction of words/numbers and punctuation

# for defaults
ocr_results_dir = None
save_binary_dir = None
make_sense_dir = None
images_jpeg_dir = None
full_article_pdfs_dir = None
make_splits = True

# final test set
save_binary_dir = '/Users/jillnaiman/MegaYolo_test/'
make_sense_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Annotations/MakeSenseAnnotations_test/'
binaries_file = 'model12_finaltest'# for final test set


# # For non-defaults (like for benchmarking), set to None for default
# ocr_results_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/OCR_processing_pmcnoncom/'
# use_pdfmining = False
# generate_features = False
# save_binary_dir = '/Users/jillnaiman/MegaYolo_pmcnoncom/'
# make_sense_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Annotations_pmcnoncom/MakeSenseAnnotations/'
# images_jpeg_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Pages_pmcnoncom/RandomSingleFromPDFIndexed/'
# full_article_pdfs_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/data/PMC_noncom/pdfs/'
# make_splits = False





# # this supercedes what is in the config file
# feature_list = ['grayscale','fontsize','carea boxes','paragraph boxes','fraction of numbers in a word','fraction of letters in a word',
#                 'punctuation','x_ascenders','x_decenders','text angles', 'word confidences','Spacy POS','Spacy TAGs','Spacy DEPs']



# feature_list = ['grayscale']
# # call these something new?
# binaries_file = 'model1_tfrecordz'

# feature_list = ['grayscale','fontsize']
# # call these something new?
# binaries_file = 'model2_tfrecordz'


# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders']
# # call these something new?
# binaries_file = 'model3_tfrecordz'


# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences']
# # call these something new?
# binaries_file = 'model4_tfrecordz'


# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation']
# # call these something new?
# binaries_file = 'model5_tfrecordz'

# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles']
# # call these something new?
# binaries_file = 'model6_tfrecordz'

# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS']
# # call these something new?
# binaries_file = 'model7_tfrecordz'


# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS','Spacy TAGs','Spacy DEPs']
# # call these something new?
# binaries_file = 'model8_tfrecordz'

# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS','Spacy TAGs','Spacy DEPs', 'paragraph boxes']
# # call these something new?
# binaries_file = 'model9_tfrecordz'


# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS','Spacy TAGs','Spacy DEPs', 'paragraph boxes', 'carea boxes']
# # call these something new?
# binaries_file = 'model10_tfrecordz'

# # hopefully this is our final model??
# feature_list = ['grayscale','fontsize', 'word confidences', 'text angles','Spacy POS']
# # call these something new?
# binaries_file = 'model11_tfrecordz'


# MODEL 12 IS SO FAR WINNING
# Start taking things out from the main list of model 7
# feature_list = ['grayscale','fontsize','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS']
# feature_list = ['grayscale','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word',
#                 'fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS']
# # call these something new?
# #binaries_file = 'model12_tfrecordz'
# binaries_file = 'model12_tfrecordz_pmcnoncom' # benchmark



# feature_list = ['grayscale','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word','punctuation', 
#                'text angles','Spacy POS']
# # call these something new?
# binaries_file = 'model13_tfrecordz'

# feature_list = ['grayscale','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word',
#                 'fraction of letters in a word', 
#                'text angles','Spacy POS']
# # call these something new?
# binaries_file = 'model14_tfrecordz'

# feature_list = ['grayscale','x_ascenders','x_decenders', 'word confidences', 
#                 'fraction of numbers in a word',
#                 'fraction of letters in a word', 'punctuation',
#                'text angles']
# # call these something new?
# binaries_file = 'model15_tfrecordz'

# # NEW model 11
# feature_list = ['grayscale','fontsize', 'word confidences', 
#                 'fraction of numbers in a word','fraction of letters in a word','punctuation', 
#                'text angles','Spacy POS']
# # call these something new?
# binaries_file = 'model11_tfrecordz'


# MODEL 12 -- winner winner, chicken dinner
feature_list = ['grayscale','x_ascenders','x_decenders', 'word confidences', 
                'fraction of numbers in a word',
                'fraction of letters in a word','punctuation', 
               'text angles','Spacy POS']
#binaries_file = 'model12_tfrecordz' # default name
#binaries_file = 'model12_finaltest'# for final test set
# # call these something new?
# binaries_file = 'model12_tfrecordz_pmcnoncom' # benchmark

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
import pickle

from annotation_utils import get_all_ocr_files, make_ann_directories, collect_ocr_process_results, \
   get_makesense_info_and_years, get_years, get_cross_index, get_pdffigures_info, get_annotation_name, \
   true_box_caption_mod

from ocr_and_image_processing_utils import angles_results_from_ocr

from feature_generation_utils import generate_single_feature

from general_utils import parse_annotation

from post_processing_utils import parse_annotations_to_labels


# general debug
debug = False


# ----------------------------------------------

if images_jpeg_dir is None: images_jpeg_dir = config.images_jpeg_dir
if save_binary_dir is None: save_binary_dir = config.save_binary_dir

# let's get all of the ocr files
ocrFiles = get_all_ocr_files(ocr_results_dir=ocr_results_dir)
# get important quantities from these files
if yt.is_root(): print('retreiving OCR data, this can take a moment...')
ws, paragraphs, squares, html, rotations,colorbars = collect_ocr_process_results(ocrFiles)
# create dataframe
df = pd.DataFrame({'ws':ws, 'paragraphs':paragraphs, 'squares':squares, 
                   'hocr':html, 'rotation':rotations, 'colorbars':colorbars})#, 'pdfwords':pdfwords})
df = df.drop_duplicates(subset='ws')
df = df.set_index('ws')

#import sys; sys.exit()

binaries_file2 = save_binary_dir + 'binaries'
if len(binaries_file)>0: #add
    binaries_file = binaries_file2 + '_' + binaries_file + '/'
else:
    binaries_file = binaries_file2 + '/'

def create_stuff(lock):
    if os.path.isfile(save_binary_dir + 'done'): os.remove(save_binary_dir + 'done') # test to make sure we don't move on in parallel too soon
    if yt.is_root():
        # check these all exist, but don't over write the directories like for annotaitons
        if not os.path.exists(binaries_file):
            os.mkdir(binaries_file)  
        # ... unless tfrecords -- don't want to keep writing
        if config.astype == 'tfrecord':
            # remove, remake
            shutil.rmtree(binaries_file)
            os.mkdir(binaries_file)
        # create records files and delete others
        if not os.path.exists(config.tmp_storage_dir+'TMPTFRECORD/'):
            os.mkdir(config.tmp_storage_dir+'TMPTFRECORD/')
        # remove, remake
        shutil.rmtree(config.tmp_storage_dir+'TMPTFRECORD/')
        os.mkdir(config.tmp_storage_dir+'TMPTFRECORD/')
        # done
        with open(save_binary_dir + 'done','w') as ffd:
            print('done!',file=ffd)
        print(save_binary_dir + 'done')

# in theory, this should stop the parallel stuff until the folder
#. has been created, but I'm not 100% sure on this one
my_lock = Lock()
create_stuff(my_lock)

# get annotations
imgDirAnn = save_binary_dir + config.ann_name + str(int(config.IMAGE_H)) + 'x' + str(int(config.IMAGE_W))  + '_ann/'
# get all annotations
annotations = glob(imgDirAnn+'*.xml')

# storage
my_storage = {}
wsInds = np.linspace(0,len(annotations)-1,len(annotations)).astype('int')
# debug
#wsInds = wsInds[:2]
mod_output = 100
                   
LABELS, labels, slabels, \
  CLASS, annotations, Y_full, maxboxes = parse_annotations_to_labels(imgDirAnn, 
                                                           '', 
                                                           benchmark=True,
                                                          return_max_boxes=True)
if yt.is_root():
    print('maximum number of boxes = ',maxboxes), 

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
    feature_name, font = generate_single_feature(dfsingle, LABELS,
                                                 maxboxes=maxboxes, 
                                           feature_list = feature_list, 
                                           binary_dir = binaries_file, 
                                           mode=mode, maxTag=maxTag,
                                                 images_jpeg_dir = images_jpeg_dir)
    
    #import sys; sys.exit()
    sto.result = [font,feature_name]

    
# if was records file, do a conversion
if yt.is_root():
    # combine fonts
    fontsize = []; names = []
    for ns,vals in sorted(my_storage.items()):
        if vals is not None:
            fontsize.append(vals[0])
            names.append(vals[1])

    # if not os.path.exists(binaries_file[:-1]+'_fonts/'):
    #     os.mkdir(binaries_file[:-1]+'_fonts/')
    # # remove, remake
    # shutil.rmtree(binaries_file[:-1]+'_fonts/')
    # os.mkdir(binaries_file[:-1]+'_fonts/')
    with open(binaries_file[:-1]+'_fonts.pickle', 'wb') as ff:
        pickle.dump([fontsize,names], ff)   
    
    if 'TMPTFRECORD/' in feature_name: # we have done temp storage
        import tensorflow as tf
        # note: y_* aren't actually used anywhere
        if splits_directory is None:
            print('not totally implemented yet!!!!')
            import sys; sys.exit()
            # but it would go something like...
#            train_per = config.train_per # ...
#             X_train, y_train, X_valid, y_valid,\
#                X_test, y_test = train_test_valid_split(X_full, Y_full,
#                                                        train_size = train_per, 
#                                                        valid_size = valid_per, 
#                                                        test_size = test_per, 
#                                                        textClassification=True, 
#                                                        asInts=False)

#             print('We have AT LEAST', len(X_train), 'training,', 
#                   len(X_valid), 'validation,', 
#                   len(X_test), 'test instances.')

#             # write files for splits
#             np.savetxt(splitsDir + 'train.csv', X_train, fmt='%s', delimiter=',')
#             np.savetxt(splitsDir + 'test.csv', X_test, fmt='%s', delimiter=',')
#             np.savetxt(splitsDir + 'valid.csv', X_valid, fmt='%s', delimiter=',')
            
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        
        # first off, save labels as CSV
        np.savetxt(binaries_file + 'LABELS.csv', LABELS, fmt='%s', delimiter=',')    
        
        # save feature list
        with open(binaries_file +'feature_list.pickle', 'wb') as ff:
            pickle.dump([feature_list], ff)   
                    
        # Create a dictionary with features that may be relevant.
        def image_example(image, boxes, img_name):
            #image_shape = tf.io.decode_jpeg(image_string).shape
            image_string = image.astype('float32')/255.0
            image_string = image.reshape(image.shape[0]*image.shape[1]*image.shape[2])

            nfeatures = image.shape[2]
            nboxes = boxes.shape[0]
            if nboxes>0:
                boxout = boxes.reshape(boxes.shape[0]*boxes.shape[1])
            else:
                boxout = np.array([])

            feature = {
              'nbox': _float_feature(np.float32(nboxes)),
              'nfeatures': _float_feature(np.float32(nfeatures)),
              'boxes': _bytes_feature(boxout.astype('float32').tobytes()),
              'image_raw': _bytes_feature(image_string.astype('float32').tobytes()),
              'image_name': _bytes_feature(img_name.encode('utf-8')),
            }
            return tf.train.Example(features=tf.train.Features(feature=feature))  
        
        classDir_main_to = save_binary_dir + config.ann_name + \
           str(int(config.IMAGE_H)) + 'x' + \
           str(int(config.IMAGE_W))  + '_ann/'
        
        classDir_main_to_imgs = save_binary_dir + feature_name.split('/')[-2] + '/'  
        
        # make a temp record file to see how big each file is, on avearge
        # write one image file and see how big it is
        record_file = config.tmp_storage_dir+'TMPTFRECORD/test.tfrecords'
        compress = 'GZIP'
        tf_record_options = tf.io.TFRecordOptions(compression_type = compress) 
        
        #with tf.io.TFRecordWriter(record_file) as writer:
        with tf.io.TFRecordWriter(record_file, options=tf_record_options) as writer:
            # make sure file is there
            success = False
            ia=0
            while not success:
                a = classDir_main_to + annotations[ia].split('/')[-1]
                try:
                #if True:
                    imgs_name, bbox = parse_annotation([a], LABELS,
                                                           feature_dir=config.tmp_storage_dir+'TMPTFRECORD/',
                                                           annotation_dir=classDir_main_to) 
                    arr = np.load(imgs_name[0])['arr_0']
                    success = True
                except:
                    print('no', a,ia)
                    ia+=1
            # fake boxes
            fakebox = np.random.random([maxboxes,5])
            tf_example = image_example(arr,fakebox,imgs_name[0])
            writer.write(tf_example.SerializeToString())
            
        # optimize to ~100Mb a file -- https://docs.w3cub.com/tensorflow~guide/performance/performance_guide
        filesize = os.path.getsize(record_file)
        #i.e we want:
        nfiles_per_file = 100*1e6//filesize
        # downgrade for compression
        #ndiv = 100.0 # about 1-2Mb/file
        #ndiv = 10.0 # about 34Mb/file
        ndiv = 4.0 # maybe ~85Mb/file (?)
        if compress is not None:
            nfiles_per_file = nfiles_per_file/ndiv
        print('there will be', nfiles_per_file, 'images+labels per TFrecord')
        
        
        if make_splits: # if we have a splits list
            splitsnames = ['train','valid','test']
        else: # or all together
            splitsnames = ['record']
        for sp in splitsnames:
            print('-----------', sp, '--------------')
            if make_splits:
                filelist1 = pd.read_csv(config.tmp_storage_dir+sp+'.csv',
                                   names=['filename'])['filename'].values
            else:
                filelist1 = glob(imgDirAnn+'*xml')
            # check for empty files
            filelist = []
            for a in filelist1:
                a = classDir_main_to + a.split('/')[-1]
                try:
                    imgs_name, bbox = parse_annotation([a], LABELS,
                                                       feature_dir=config.tmp_storage_dir+'TMPTFRECORD/',
                                                       annotation_dir=classDir_main_to) 
                    filelist.append(imgs_name[0])
                except:
                    print('no file', a)    
                    
            nfiles = int(np.ceil(len(filelist)*1.0/nfiles_per_file))
            itrain = 0
            record_file = binaries_file+sp+'_{}.tfrecords'

            # if I was clever I would do this in parallel but...
            itotalLoop = 0
            for index in range(nfiles):
                if index%1 == 0: print('on', index,'of',nfiles)
                with tf.io.TFRecordWriter(record_file.format(index), options=tf_record_options) as writer:
                    for iloop,a in enumerate(filelist[index*int(nfiles_per_file):min([(index+1)*int(nfiles_per_file),len(filelist)])]):
                        #print(iloop,a)
                        a = classDir_main_to + a.split('/')[-1].split('.npz')[0] + '.xml'

                        try:
                            imgs_name, bbox = parse_annotation([a], LABELS,
                                                               feature_dir=config.tmp_storage_dir+'TMPTFRECORD/',
                                                               annotation_dir=classDir_main_to)
                        except:
                            print('STILL cant find', a, ', moving on...')
                            continue
                        arr = np.load(imgs_name[0])['arr_0']

                        if len(bbox) > 0: 
                            bbox = np.array(bbox[0])
                        else:
                            bbox = np.array([])
                        tf_example = image_example(arr,bbox,imgs_name[0])
                        writer.write(tf_example.SerializeToString())
                        
        # remove all tmp files
        shutil.rmtree(config.tmp_storage_dir+'TMPTFRECORD/')
        print('All done!')
            