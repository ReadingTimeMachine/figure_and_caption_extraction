# set to None if you want to use defaults
# binary_dirs = 'binaries_model1/'
# weightsFileDir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/saved_weights/20211111_model1/'
# weightsFile = 'training_1model1_model_l0.17215717.h5' # figure/table, fig/table captions

# binary_dirs = 'binaries_model2/'
# weightsFileDir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/saved_weights/20211116_model2/'
# weightsFile = 'training_1model2_model_l0.13648733.h5' # figure/table, fig/table captions

# binary_dirs = 'binaries_model3/'
# weightsFileDir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/saved_weights/20211117_model3/'
# weightsFile = 'training_1model3_model_l0.20664357.h5' # figure/table, fig/table captions

# weightsFileDir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/saved_weights/20211118_model4/'
# binary_dirs = 'binaries_model4/'
# weightsFile = 'training_1model4_model_l0.1202761.h5' # figure/table, fig/table captions

binary_dirs = 'binaries_model5/'
weightsFileDir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/saved_weights/20211124_model5/'
weightsFile = 'training_1model5_model_l0.13795476.h5' # figure/table, fig/table captions



#adder = '_mod1' # leave empty to save default file
adder = '' # leave empty to save default file

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
yt.enable_parallelism()
import pandas as pd
import pickle
import numpy as np
from annotation_utils import get_all_ocr_files, collect_ocr_process_results, \
   get_makesense_info_and_years, get_years
from post_processing_utils import parse_annotations_to_labels, build_predict, \
   get_true_boxes, get_ocr_results, get_image_process_boxes, clean_overlapping_squares, \
   clean_merge_pdfsquares, clean_merge_heurstic_captions, add_heuristic_captions, \
   clean_found_overlap_with_ocr, clean_true_overlap_with_ocr, clean_merge_squares, \
   clean_big_captions, clean_match_fig_cap, expand_true_boxes_fig_cap, \
   expand_found_boxes_fig_cap, expand_true_area_above_cap, expand_found_area_above_cap
#, calc_metrics
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
# # set up annotations CV arrays -- this will calculate spread around metrics
# inds = np.random.choice(range(len(annotations)),size=len(annotations), replace=False)
# # split into k-cv indicies
# ann_inds = []
# incr = len(annotations)//n_folds_cv
# for ix in range(n_folds_cv):
#     for inc in range(incr):
#         ann_inds.append(ix)
# while len(inds) > len(ann_inds): # while not perfectly split, add extras to random fold
#     ann_inds.append(np.random.randint(n_folds_cv))
# if len(inds) != len(ann_inds): print('issue here!!!'); import sys; sys.exit()
# sortp = np.argsort(inds)
# ann_inds = np.array(ann_inds)[sortp]

wsInds = np.arange(0,len(annotations))
#wsInds = np.arange(0,6) # debug
#wsInds = wsInds[1:]
#wsInds = wsInds[:2]

# run the thing
iMod = 10

for sto, icombo in yt.parallel_objects(wsInds, config.nProcs, storage=my_storage):
    
    a = annotations[icombo] # which annotation
    #k_cv = ann_inds[icombo] # which fold?

    # run model
    if icombo%iMod == 0:
        print('on ', icombo, ' of ', len(annotations)-1)
        
    # there is a lot of mess here that gets and formats all true boxes and 
    #. all of the OCR data
    imgs_name, pdfboxes, pdfrawboxes,years_ind, truebox = get_true_boxes(a,LABELS,
                                                       badskews,badannotations,
                                                       annotation_dir=annotation_dir,
                                                      feature_dir=feature_dir)
    
    #import sys; sys.exit()
    
    # get OCR results and parse them, open image for image processing
    backtorgb,image_np,rotatedImage,rotatedAngleOCR,bbox_hocr,\
      bboxes_words,bbsq,rotation,bbox_par = get_ocr_results(imgs_name, dfMakeSense,df)
    
    # predict squares in 2 ways
    # 1. MEGA YOLO
    boxes, scores, labels = model.predict(image_np[np.newaxis, ...])
    boxes1, scores1, labels1 = np.squeeze(boxes, 0), np.squeeze(scores, 0), np.squeeze(labels, 0)

    #save_boxes = boxes.copy(); save_labels = labels.copy(); save_scores2 = scores.copy()

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
    
    # ------------------
    
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
    # boxes_heur2, labels_heur2, scores_heur2 = add_heuristic_captions(bbox_figcap_pars,
    #                                                               captionText_figcap,
    #                                                               ibbOverlap,
    #                                                               boxes_heur, 
    #                                                               labels_heur, 
    #                                                               scores_heur, dfMS)
    
    # clean found boxes by paragraphs and words  -- if found box overlaps with 
    #. an OCR box, include this box in the bounding box of captions
    # boxes_par_found, labels_par_found, \
    #   scores_par_found = clean_found_overlap_with_ocr(boxes_heur2, labels_heur2, 
    #                                             scores_heur2,bboxes_words,
    #                                                   bbox_par,rotation,
    #                                                   LABELS, dfMS)  
    # other way -- w/o adding more heursitic caps:
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
    boxes_sq2, labels_sq2, scores_sq2 = clean_big_captions(boxes_sq1,
                                                        labels_sq1,
                                                        scores_sq1, 
                                                        LABELS)
    # --- 6 ---
    # sometimes captions are slightly overlapping with figures -- split the 
    # difference between those where they touch on the "bottom"
    # Default to captions found with mega yolo, if there is a figure but 
    #. no caption found, then see if there is a heuristically found caption
    boxes_sq3, labels_sq3, scores_sq3 = clean_match_fig_cap(boxes_sq2,
                                                             labels_sq2,
                                                         scores_sq2, bbsq, 
                                                         LABELS, 
                                                         rotatedImage, 
                                                         rotatedAngleOCR,
                                                         dfMS)
    
    # check for overlaps
    truebox2 = expand_true_boxes_fig_cap(truebox1, rotatedImage, LABELS)
    # again for found boxes?  I feel like maybe not the one above?
    boxes_sq4, labels_sq4, scores_sq4 = expand_found_boxes_fig_cap(boxes_sq3, 
                                                                labels_sq3, 
                                                                scores_sq3,
                                                                   bbsq,
                                                                rotatedImage, 
                                                                LABELS, dfMS)
    
    # expand true boxes if area above caption is larger
    truebox3 = expand_true_area_above_cap(truebox2, rotatedImage, LABELS)
    # same for found
    boxes_sq5, labels_sq5, scores_sq5 = expand_found_area_above_cap(boxes_sq4, 
                                                                    labels_sq4, 
                                                                    scores_sq4, 
                                                                    bbsq,
                                                                    rotatedImage, 
                                                                    LABELS, dfMS)
    
    sto.result_id = icombo
    #if icombo==1: import sys; sys.exit()
    sto.result = [icombo,imgs_name[0], truebox, pdfboxes, pdfrawboxes, captionText_figcap, 
                  bbox_figcap_pars,
                  sboxes_cleaned, slabels_cleaned, sscores_cleaned, 
                 boxes_pdf, labels_pdf, scores_pdf, 
                  boxes_heur, labels_heur, scores_heur,
                 boxes_heur2, labels_heur2, scores_heur2,
                 boxes_par_found, labels_par_found, scores_par_found,
                 boxes_sq1, labels_sq1, scores_sq1,
                 boxes_sq2, labels_sq2, scores_sq2,
                 boxes_sq3, labels_sq3, scores_sq3,
                 boxes_sq4, labels_sq4, scores_sq4,
                 boxes_sq5, labels_sq5, scores_sq5,
                 truebox1,truebox2,truebox3,rotatedImage,LABELS, boxes1, scores1, labels1]
    
    
if yt.is_root():
    icombo,imgs_name, truebox, pdfboxes, pdfrawboxes, captionText_figcap = [],[],[],[],[],[]
    bbox_figcap_pars = []
    sboxes_cleaned, slabels_cleaned, sscores_cleaned = [],[],[]
    boxes_pdf, labels_pdf, scores_pdf = [], [],[]
    boxes_heur, labels_heur, scores_heur = [], [], []
    boxes_heur2, labels_heur2, scores_heur2 = [],[],[]
    boxes_par_found, labels_par_found, scores_par_found = [],[],[]
    boxes_sq1, labels_sq1, scores_sq1 = [],[],[]
    boxes_sq2, labels_sq2, scores_sq2 = [],[],[]
    boxes_sq3, labels_sq3, scores_sq3 = [],[],[]
    boxes_sq4, labels_sq4, scores_sq4 = [],[],[]
    boxes_sq5, labels_sq5, scores_sq5 = [],[],[]
    truebox1,truebox2,truebox3,rotatedImage,LABELS = [],[],[],[],[]
    boxes1, scores1, labels1 = [],[],[]
    
    for ns,vals in sorted(my_storage.items()):
        if vals is not None:
            icombo.append(vals[0])
            imgs_name.append(vals[1])
            truebox.append(vals[2])
            pdfboxes.append(vals[3])
            pdfrawboxes.append(vals[4])
            captionText_figcap.append(vals[5])
            bbox_figcap_pars.append(vals[6])
            sboxes_cleaned.append(vals[7])
            slabels_cleaned.append(vals[8])
            sscores_cleaned.append(vals[9])
            boxes_pdf.append(vals[10])
            labels_pdf.append(vals[11])
            scores_pdf.append(vals[12])
            boxes_heur.append(vals[13])
            labels_heur.append(vals[14])
            scores_heur.append(vals[15])
            boxes_heur2.append(vals[16])
            labels_heur2.append(vals[17])
            scores_heur2.append(vals[18])
            boxes_par_found.append(vals[19])
            labels_par_found.append(vals[20])
            scores_par_found.append(vals[21])
            boxes_sq1.append(vals[22])
            labels_sq1.append(vals[23])
            scores_sq1.append(vals[24])
            boxes_sq2.append(vals[25])
            labels_sq2.append(vals[26])
            scores_sq2.append(vals[27])
            boxes_sq3.append(vals[28])
            labels_sq3.append(vals[29])
            scores_sq3.append(vals[30])
            boxes_sq4.append(vals[31])
            labels_sq4.append(vals[32])
            scores_sq4.append(vals[33])
            boxes_sq5.append(vals[34])
            labels_sq5.append(vals[35])
            scores_sq5.append(vals[36])
            truebox1.append(vals[37])
            truebox2.append(vals[38])
            truebox3.append(vals[39])
            rotatedImage.append(vals[40])
            LABELS.append(vals[41])
            boxes1.append(vals[42])
            scores1.append(vals[43])
            labels1.append(vals[44])
            
    # update labels
    LABELS = LABELS[0]
            
# binary_dirs = 'binaries_model1/'
# weightsFileDir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/saved_weights/20211111_model1/'
# weightsFile = 'training_1model1_model_l0.17215717.h5' # figure/table, fig/table captions
            
    # build up filename
    pp = config.metric_results_dir
    pp += binary_dirs.split('/')[0]
    pp += adder
    pp += '.pickle'
    with open(pp, 'wb') as ff:
        pickle.dump([icombo,imgs_name, truebox, pdfboxes, pdfrawboxes, captionText_figcap,\
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
                     truebox1,truebox2,truebox3,rotatedImage,LABELS,boxes1, scores1, labels1], ff)
            
