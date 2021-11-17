# set to None if you want to use defaults
binary_dirs = 'binaries_model1/'
weightsFileDir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/saved_weights/20211111_model1/'
weightsFile = 'training_1model1_model_l0.17215717.h5' # figure/table, fig/table captions

benchmark = None
scoreminVec = None
iouminVec = None



import config



# from config file
annotation_dir = config.save_binary_dir + config.ann_name + str(config.IMAGE_H) + 'x' + str(config.IMAGE_W) + '_ann/'

#'/Users/jillnaiman/MegaYolo/yolo_512x512_ann/'

if binary_dirs is None: binary_dirs = 'binaries/'

#classDir_main_to_imgs = config.save_binary_dir + binary_dirs
feature_dir = config.save_binary_dir + binary_dirs

pickle_dir = config.ocr_results_dir
makeSenseDir = config.make_sense_dir
images_pulled_dir = config.images_jpeg_dir
badskewList = config.make_sense_dir+config.bad_skews_file
badannotationsList = badskewList # if 2 different lists
ocrFilesAll = [config.ocr_results_dir + config.pickle_file_head + '*.pickle']
n_folds_cv = config.n_folds_cv
if weightsFileDir is None: weightsFileDir = config.weightsFileDir
if weightsFile is None: weightsFile = config.weightsFile
if benchmark is None: benchmark = config.benchmark

# make pickle file name
yolopicklename = 'mega_yolov5_' + binary_dirs.split('/')[-2] +'_'+ weightsFileDir.split('/')[-2] + '.pickle'

version = config.version 

if scoreminVec is None: scoreminVec = config.scoreminVec
if iouminVec is None: iouminVec = config.iouminVec

diagnostics_dir = config.tmp_storage_dir
store_diagnostics = config.store_diagnostics

weightsFileDownload = weightsFileDir + weightsFile
if not benchmark:
    testListFile = weightsFileDir + 'testList.csv'
anchorsFile = weightsFileDir + 'anchors.pickle'


#################################################
import yt
import pandas as pd
import pickle
import numpy as np
from annotation_utils import get_all_ocr_files, collect_ocr_process_results, \
   get_makesense_info_and_years, get_years
from post_processing_utils import parse_annotations_to_labels, build_predict, \
   get_true_boxes, get_ocr_results, get_image_process_boxes, clean_overlapping_squares, \
   clean_merge_pdfsquares, clean_merge_heurstic_captions, add_heuristic_captions, \
   clean_found_overlap_with_ocr, clean_true_overlap_with_ocr, clean_merge_squares, \
   clean_big_captions, clean_match_fig_cap
#################################################

if store_diagnostics:
    # remove any files in subfolders
    for f in os.listdir(diagnostics_dir + 'FN/'):
        os.remove(os.path.join(diagnostics_dir + 'FN/', f))
    for f in os.listdir(diagnostics_dir + 'FP/'):
        os.remove(os.path.join(diagnostics_dir + 'FP/', f))
    for f in os.listdir(diagnostics_dir + 'TP/'):
        os.remove(os.path.join(diagnostics_dir + 'TP/', f))

# let's get all of the ocr files
ocrFiles = get_all_ocr_files()
# get important quantities from these files
if yt.is_root(): print('retreiving OCR data, this can take a moment...')
ws, paragraphs, squares, html, rotations = collect_ocr_process_results(ocrFiles)
# create dataframe
df = pd.DataFrame({'ws':ws, 'paragraphs':paragraphs, 'squares':squares, 
                   'hocr':html, 'rotation':rotations})#, 'pdfwords':pdfwords})
df = df.drop_duplicates(subset='ws')
df = df.set_index('ws')

# read in anchors
saveFileAnchors = weightsFileDir + 'anchors.pickle'
with open(saveFileAnchors, 'rb') as f:
    myanchors = pickle.load(f) 
    myanchors = myanchors.astype('float32')
    # don't ask
    anchors = myanchors
    

    
LABELS, labels, slabels, \
  CLASS, annotations, Y_full = parse_annotations_to_labels(annotation_dir, 
                                                           testListFile, 
                                                           benchmark=benchmark)

# checks
if yt.is_root():
    print('LABELS=', LABELS)
    #print('unique Y_full=',np.unique(Y_full), np.unique(Y_full_str))
    
# build the model
model = build_predict(weightsFileDownload, anchorsFile, 
                    feature_dir,LABELS,version='l', debug=False)
model.load_weights(weightsFileDownload)

if badskewList is not None:
    badskews = pd.read_csv(badskewList); badannotations = pd.read_csv(badannotationsList)
    badskews = badskews['filename'].values.tolist()
    badannotations = badannotations['filename'].values.tolist()
else:
    badannotations = []; badskews = []

# get make sense info
dfMakeSense = get_makesense_info_and_years(df)

# get years and years list
years, years_list = get_years(dfMakeSense['filename'].values)


my_storage = {}
# set up annotations CV arrays -- this will calculate spread around metrics
inds = np.random.choice(range(len(annotations)),size=len(annotations), replace=False)
# split into k-cv indicies
ann_inds = []
incr = len(annotations)//n_folds_cv
for ix in range(n_folds_cv):
    for inc in range(incr):
        ann_inds.append(ix)
while len(inds) > len(ann_inds): # while not perfectly split, add extras to random fold
    ann_inds.append(np.random.randint(n_folds_cv))
if len(inds) != len(ann_inds): print('issue here!!!'); import sys; sys.exit()
sortp = np.argsort(inds)
ann_inds = np.array(ann_inds)[sortp]

wsInds = np.arange(0,len(annotations))



# run the thing
iMod = 10

for sto, icombo in yt.parallel_objects(wsInds, config.nProcs, storage=my_storage):
    TPv = np.zeros([len(LABELS), len(iouminVec), len(scoreminVec)]) # true positivies in each category (each cateogry of *true* boxes)
    FPv = np.zeros([len(LABELS), len(iouminVec), len(scoreminVec)]) # false positivies, each category (each category of *found* boxes)
    FNv = np.zeros([len(LABELS), len(iouminVec), len(scoreminVec)]) # false negatives, each category (each category of *true* boxes)
    MISSCLASSv = np.zeros([len(LABELS), len(iouminVec), len(scoreminVec)]) # found, but not the right type
    THROWNOUTv = np.zeros([len(LABELS), len(iouminVec), len(scoreminVec)]) # found, but culled out
    totalTruev = np.zeros([len(LABELS), len(iouminVec), len(scoreminVec)]) # count numbers in category
    
    # by year
    TPyear = np.zeros([len(years),len(LABELS), len(iouminVec), len(scoreminVec)]) # true positivies in each category (each cateogry of *true* boxes)
    FPyear = np.zeros([len(years),len(LABELS), len(iouminVec), len(scoreminVec)]) # false positivies, each category (each category of *found* boxes)
    FNyear = np.zeros([len(years),len(LABELS), len(iouminVec), len(scoreminVec)]) # false negatives, each category (each category of *true* boxes)
    MISSCLASSyear = np.zeros([len(years),len(LABELS), len(iouminVec), len(scoreminVec)]) # found, but not the right type
    THROWNOUTyear = np.zeros([len(years),len(LABELS), len(iouminVec), len(scoreminVec)]) # found, but culled out
    totalTrueyear = np.zeros([len(years),len(LABELS), len(iouminVec), len(scoreminVec)]) # count numbers in category
    
    a = annotations[icombo] # which annotation
    k_cv = ann_inds[icombo] # which fold?

    # run model
    if icombo%iMod == 0:
        print('on ', icombo, ' of ', len(annotations)-1)
        
    # there is a lot of mess here that gets and formats all true boxes and 
    #. all of the OCR data
    imgs_name, pdfboxes, pdfrawboxes,years_ind, truebox = get_true_boxes(a,LABELS,
                                                       badskews,badannotations,
                                                       annotation_dir=annotation_dir,
                                                      feature_dir=feature_dir)
    
    # get OCR results and parse them, open image for image processing
    backtorgb,image_np,rotatedImage,rotatedAngleOCR,bbox_hocr,\
      bboxes_words,bbsq,rotation,bbox_par = get_ocr_results(imgs_name, dfMakeSense,df)
    
    # predict squares in 2 ways
    # 1. MEGA YOLO
    boxes, scores, labels = model.predict(image_np[np.newaxis, ...])
    boxes1, scores1, labels1 = np.squeeze(boxes, 0), np.squeeze(scores, 0), np.squeeze(labels, 0)

    save_boxes = boxes.copy(); save_labels = labels.copy(); save_scores2 = scores.copy()

    # only non -1 ones
    boxes1 = boxes1[labels1>-1]
    scores1 = scores1[labels1>-1]
    labels1 = labels1[labels1>-1]    
    
    # get figures and captions from image processing
    captionText_figcap, bbox_figcap_pars = get_image_process_boxes(backtorgb, 
                                                                   bbox_hocr, 
                                                                   rotatedImage)
    
    # clean overlapping squares
    # if squares are majorly overlapping, take the one with the highest score
    sboxes_cleaned, slabels_cleaned, sscores_cleaned = clean_overlapping_squares(boxes1,
                                                                                 scores1,
                                                                                 labels1,
                                                                                 imgs_name)
    
    # probably do this earlier and pass it...
    ff = imgs_name[0].split('/')[-1].split('.npz')[0]
    dfMS = dfMakeSense.loc[dfMakeSense['filename']==ff]

    
    # merge with any boxes that have been found with PDF mining
    # found figures are generally not accurate, so ignore these, but do 
    # assume any tables or figure captions are more accurate from PDF mining
    boxes_pdf, labels_pdf, scores_pdf = clean_merge_pdfsquares(pdfboxes,
                                                               pdfrawboxes,
                                                               sboxes_cleaned, 
                                                               slabels_cleaned, 
                                                               sscores_cleaned, 
                                                               LABELS, dfMS)
    
    # combine figure caption boxes with heuristically found ones
    # -- often the heurstically found boxes are more accurate, especially 
    # in the vertical direction
    boxes_heur, labels_heur, scores_heur,\
      ibbOverlap = clean_merge_heurstic_captions(boxes_pdf, 
                                            labels_pdf, scores_pdf, 
                                            bbox_figcap_pars, LABELS,dfMS)
     
        
        
    # sometimes figures are found, but no captions -- check for "extra" 
    # only heuristically found captions, and use these as a last resort
    # when matching figures to captions
    boxes_heur, labels_heur, scores_heur = add_heuristic_captions(bbox_figcap_pars,
                                                                  captionText_figcap,
                                                                  ibbOverlap,
                                                                  boxes_heur, 
                                                                  labels_heur, 
                                                                  scores_heur, dfMS)
    
    # clean found boxes by paragraphs and words  -- if found box overlaps with 
    #. an OCR box, include this box in the bounding box of captions
    boxes_par_found, labels_par_found, \
      scores_par_found = clean_found_overlap_with_ocr(boxes_heur, labels_heur, 
                                                scores_heur,bboxes_words,
                                                      bbox_par,rotation,
                                                      LABELS, dfMS)
    
    # do same excersize with trueboxes (already done really in processing annoations)
    truebox1 = clean_true_overlap_with_ocr(truebox, bboxes_words,
                                           bbox_par,rotation, 
                                           LABELS, dfMS)
    
    # if figure boxes are smaller than image-processing found boxes, merge them; 
    boxes_sq1, labels_sq1, scores_sq1 = clean_merge_squares(bbsq, 
                                                            boxes_par_found, 
                                                            labels_par_found, 
                                                            scores_par_found, 
                                                            LABELS, dfMS)
    
    # if there are any huge captions -- like 75% of the area of the page or more
    #. these are wrong, so drop them
    boxes_sq, labels_sq, scores_sq = clean_big_captions(boxes_sq1,
                                                        labels_sq1,
                                                        scores_sq1, 
                                                        LABELS)
    
    # sometimes captions are slightly overlapping with figures -- split the 
    # difference between those where they touch on the "bottom"
    # Default to captions found with mega yolo, if there is a figure but 
    #. no caption found, then see if there is a heuristically found caption
    boxes_sq, labels_sq, scores_sq = clean_match_fig_cap(boxes_sq,labels_sq,
                                                         scores_sq, bbsq, 
                                                         LABELS, 
                                                         rotatedImage, 
                                                         rotatedAngleOCR,
                                                         dfMS)
    
    
    import sys; sys.exit()