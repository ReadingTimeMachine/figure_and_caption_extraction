# process MakeSense.ai into annotations and features for mega-yolo training
import config

pdffigures_dpi = 72 # this is the default DPI of coordinates for PDFs for pdffigures2 (docs say 150, but this is a LIE)
reRun = False # only toggle on if you want to re-run all of pdffigures2 which can take a while
use_pdfmining = True # generally true, but can set to false for some benchmarks
generate_features = False # again, generally true, but set to false if you don't want to generate features

# for defaults
ocr_results_dir = None
save_binary_dir = None
make_sense_dir = None
images_jpeg_dir = None
full_article_pdfs_dir = None

plot_diagnostics = False

# For non-defaults (like for benchmarking), set to None for default

# # PMC PubLayNet
# ocr_results_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/OCR_processing_pmcnoncom/'
# use_pdfmining = False
# generate_features = False
# save_binary_dir = '/Users/jillnaiman/MegaYolo_pmcnoncom/'
# make_sense_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Annotations_pmcnoncom/MakeSenseAnnotations/'
# images_jpeg_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Pages_pmcnoncom/RandomSingleFromPDFIndexed/'
# full_article_pdfs_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/data/PMC_noncom/pdfs/'

# # ScanBank
# ocr_results_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/OCR_processing_scanbank/'
# use_pdfmining = True
# generate_features = False
# save_binary_dir = '/Users/jillnaiman/MegaYolo_scanbank/'
# make_sense_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Annotations_scanbank/MakeSenseAnnotations/'
# images_jpeg_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Pages_scanbank/RandomSingleFromPDFIndexed/'
# full_article_pdfs_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/data/scanbank/etds/'

# Final Test Dataset
ocr_results_dir = None #'/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/OCR_processing/'
use_pdfmining = True
generate_features = False
save_binary_dir = '/Users/jillnaiman/MegaYolo_test/'
make_sense_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Annotations/MakeSenseAnnotations_test/'
images_jpeg_dir = None #'/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Pages_scanbank/RandomSingleFromPDFIndexed/'
full_article_pdfs_dir = None #'/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/data/scanbank/etds/'


# ----------------------------------------------

# colors for diagnostic plots
colfig_orig = (0,0,255)
colfig_mod = (0,255,255)

colcap_orig = (255,0,0)
colcap_mod = (255,0,255)

coltab = (0,255,0) # no change
colmath = (255,255,0) # no change


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
import xml.etree.ElementTree as ET

from annotation_utils import get_all_ocr_files, make_ann_directories, collect_ocr_process_results, \
   get_makesense_info_and_years, get_years, get_cross_index, get_pdffigures_info, get_annotation_name, \
   true_box_caption_mod

from ocr_and_image_processing_utils import angles_results_from_ocr

from feature_generation_utils import generate_single_feature

from general_utils import parse_annotation

# general debug
debug = False

# ----------------------------------------------

if full_article_pdfs_dir is None: full_article_pdfs_dir = config.full_article_pdfs_dir

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

# make all file directories
if save_binary_dir is None: 
    fileStorage = config.save_binary_dir
else:
    fileStorage = save_binary_dir
imgDir, imgDirAnn, imgDirPDF, badskewsList = make_ann_directories(save_binary_dir=save_binary_dir, 
                                                                 make_sense_dir=make_sense_dir)

if plot_diagnostics is None: plot_diagnostics = config.plot_diagnostics
# for saving diagnostics, if you've chosen to do that
diagnostics_file = config.tmp_storage_dir + 'tmpAnnDiags/'

if plot_diagnostics:
    from PIL import Image


def create_stuff(lock):
    if os.path.isfile(fileStorage + 'done'): os.remove(fileStorage + 'done') # test to make sure we don't move on in parallel too soon
    if yt.is_root():
        # check these all exist, but don't over write the directories like for annotaitons
        if not os.path.exists(fileStorage):
            os.mkdir(fileStorage)
        if not os.path.exists(fileStorage+'binaries/'):
            os.mkdir(fileStorage+'binaries/')  
            
        # delete all, remake
        if not os.path.exists(imgDirAnn):
            os.makedirs(imgDirAnn)
        shutil.rmtree(imgDirAnn)
        os.makedirs(imgDirAnn)
        # if pdf things not there, then make it, but don't delete it if its there
        if not os.path.exists(imgDirPDF):
            os.makedirs(imgDirPDF)

        if plot_diagnostics:
            if not os.path.exists(diagnostics_file): # main file
                os.makedirs(diagnostics_file)
            ## delete, remake
            shutil.rmtree(diagnostics_file)
            os.makedirs(diagnostics_file)
        # done
        with open(fileStorage + 'done','w') as ffd:
            print('done!',file=ffd)
        print(fileStorage + 'done')

# in theory, this should stop the parallel stuff until the folder
#. has been created, but I'm not 100% sure on this one
my_lock = Lock()
create_stuff(my_lock)


# get make sense info
dfMakeSense = get_makesense_info_and_years(df,make_sense_dir=make_sense_dir)
LABELS = []
for s in dfMakeSense['squares'].values:
    for ss in s:
        LABELS.append(ss[-1])
LABELS = np.unique(LABELS).tolist()

# get years and years list
years, years_list = get_years(dfMakeSense['filename'].values)

# storage
my_storage = {}
wsInds = np.linspace(0,len(dfMakeSense['filename'].values)-1,len(dfMakeSense['filename'].values)).astype('int')
# debug
#wsInds = wsInds[:2]
mod_output = 100

#import sys; sys.exit()

# lets do this thing...
if yt.is_root(): print('Making annotation files and features...')

for sto, iw in yt.parallel_objects(wsInds, config.nProcs, storage=my_storage):
    if iw%mod_output == 0: print('On ' + str(iw) + ' of ' + str(len(dfMakeSense['filename'].values)))

    sto.result_id = iw
        
    img_resize=(config.IMAGE_H, config.IMAGE_W)
    
    # subset dataframe
    d = dfMakeSense.loc[dfMakeSense['filename']==dfMakeSense['filename'].values[iw]]

    # get squares & save -- check for any NotSures and continue if found
    gotSomething = False
    scount=0;sfcount=0; ccount = 0
    labelsHere = []
    for b in d['squares'].values:
        gotSomething = True
        # count annotations of different counts -- used for changing subfig captions into captions as needed
        if b[-1] == 'figure': scount +=1
        if b[-1] == 'sub fig caption': sfcount +=1
        if b[-1] == 'figure caption': ccount +=1  
        labelsHere.append(b[-1])
    # if we have this tagged as NotSure or no label -- don't count these
    if 'NotSure' in labelsHere:
        gotSomething = False
    if not gotSomething: 
        print('NotSure for ', d['filename'].values)
        continue
    
    # cross index this makesense data frame with the OCR processing results
    # goOn lets us know if we should continue or not
    goOn, dfsingle, indh, fracxDiag, fracyDiag, fname = get_cross_index(d,df,img_resize,
                                                                       images_jpeg_dir=images_jpeg_dir)
    if not goOn: continue

    _, rotation, _, _, _, bbox_par, bboxes_words = angles_results_from_ocr(dfsingle['hocr'], 
                                                                           return_extras=True)
        
    bboxes_combined = []
    for ibb, bp in enumerate(bboxes_words): 
        bboxes_combined.append(bp)
    for ibb, bp in enumerate(bbox_par): # these are also xmin,ymin,xmax,ymax -- found w/OCR, original page size
        bboxes_combined.append(bp)
        
    # run pdffigures2 on this thing
    fileExpected = imgDirPDF + fname.split('/')[-1].split('_p')[0] + '.json'
    pdfExpected = full_article_pdfs_dir + fname.split('/')[-1].split('_p')[0] + '.pdf'
    # do we have any pdf heres?
    if os.path.exists(pdfExpected) and use_pdfmining:
        if not os.path.exists(fileExpected) or reRun:
            #import sys; sys.exit()
            try:
                subprocess.check_call(
                'java'
                ' -jar {pdffigures_jar_path}'
                ' --figure-data-prefix {pdffigures_dir}'
                ' --save-regionless-captions'
                ' {pdf_path}'.format(
                    pdffigures_jar_path=config.pdffigures_jar_path,
                    pdf_path=pdfExpected,
                    pdffigures_dir=imgDirPDF),
                shell=True)  
            except:
                if debug:
                    print('----- ERROR: something wrong has occured with pdffigures2 ------')
                    print(' on #',iw, ws[iw])
                pass
    else:
        if use_pdfmining:
            print('no PDF for', fname.split('/')[-1])

        
    # annotation file save (aloc) and feature file save (floc)
    # open up annotation folder
    aloc = imgDirAnn + d['filename'].values[0] + '.xml'
    floc = imgDir + d['filename'].values[0] + '.npz'

    # test
    if len(floc.split('/')[-1]) != len(aloc.split('/')[-1]):
        print(floc, aloc)
        print('NOPE')
        import sys; sys.exit()    
        
    # also check for in bad skews 
    if d['filename'].values[0]+'.png' in badskewsList:
        print('bad skew for', d['filename'].values[0], 'moving on...')
        continue
        
        
    # if we've made it this far, let's generate features
    if generate_features:
        feature_name = generate_single_feature(dfsingle,LABELS)#,
                                               #feature_list = feature_list, 
                                               # binary_dir = storeTmps+'binaries/',
                                               # images_jpeg_dir = images_jpeg_dir,)    
    
    # write file header
    fo = open(aloc,"w")
    sto.result = [aloc]
    fo.write("<annotation>\n")
    fo.write("\t<filename>"+floc+"</filename>\n")
    fo.write("\t<size>\n")
    fo.write("\t\t<width>" + str(config.IMAGE_W) + "</width>\n")
    fo.write("\t\t<height>" + str(config.IMAGE_H) + "</height>\n")
    fo.write("\t\t<depth>" + str(1) + "</depth>\n")
    fo.write("\t</size>\n")
    # add year info
    fo.write("\t<year>"+str(int(years_list[iw]))+"</year>\n")
    if len(years[years>-1]) > 0:
        fo.write("\t<yearMin>"+str(int(np.min(years[years>-1])))+"</yearMin>\n")
        fo.write("\t<yearMax>"+str(int(np.max(years[years>-1])))+"</yearMax>\n")
        fo.write("\t<yearMean>"+str(int(np.mean(years[years>-1])))+"</yearMean>\n")

    # get pdffigures2 info and write it as well
    jsonbase = fname.split('/')[-1].split('_p')[0]+'.json'
    jsonfile = imgDirPDF + jsonbase

    try:
        page = int(d['filename'].values[0].split('_p')[-1])
    except:
        print('Issue with finding page for:', fname)
        continue

    # note: xc will probably always be 0, but this is useful if scanned JPEGs
    #. come from a different location than the PDF pages
    if use_pdfmining:
        figsThisPage, rawBoxThisPage, xc, fracy, \
           fracyYOLO, fracxYOLO = get_pdffigures_info(jsonfile, 
                                                      page,d['filename'].values[0],
                                                      d,pdffigures_dpi=pdffigures_dpi)

        # if we have any PDF boxes, write them out:
        for fp in figsThisPage:
            x1 = fp['regionBoundary']['x1'] 
            y1 = fp['regionBoundary']['y1']
            x2 = fp['regionBoundary']['x2'] 
            y2 = fp['regionBoundary']['y2']
            # transform from PDF to scanned axis to YOLO boxsize
            y1 = int(round((y1/fracy)*fracyYOLO)); y2 = int(round((y2/fracy)*fracyYOLO))
            x1 = int(round((x1/fracy-xc)*fracxYOLO)); x2=int(round((x2/fracy-xc)*fracxYOLO))
            #print(xc)
            fo.write("\t<PDFinfo>\n")
            if fp['figType'] == 'Figure':
                fo.write("\t\t<name>"+'figure'+"</name>\n")
                capName = 'figure caption'
            elif fp['figType'] == 'Table':
                fo.write("\t\t<name>"+'table'+"</name>\n")
                capName = 'table caption'
            # write figure/table
            fo.write("\t\t<bndbox>\n")
            fo.write("\t\t\t<xmin>" + str(int(round(x1))) + "</xmin>\n")
            fo.write("\t\t\t<ymin>" + str(int(round(y1))) + "</ymin>\n")
            fo.write("\t\t\t<xmax>" + str(int(round(x2))) + "</xmax>\n")
            fo.write("\t\t\t<ymax>" + str(int(round(y2))) + "</ymax>\n")
            fo.write("\t\t</bndbox>\n")    
            fo.write("\t</PDFinfo>\n")
            # write caption
            x1 = fp['captionBoundary']['x1'] 
            y1 = fp['captionBoundary']['y1']
            x2 = fp['captionBoundary']['x2'] 
            y2 = fp['captionBoundary']['y2']
            # transform from PDF to scanned axis
            #y1 = int(round(y1/fracy)); y2 = int(round(y2/fracy))
            #x1 = int(round(x1/fracy-xc)); x2=int(round(x2/fracy-xc))
            y1 = int(round((y1/fracy)*fracyYOLO)); y2 = int(round((y2/fracy)*fracyYOLO))
            x1 = int(round((x1/fracy-xc)*fracxYOLO)); x2=int(round((x2/fracy-xc)*fracxYOLO))
            fo.write("\t<PDFinfo>\n")
            fo.write("\t\t<name>"+capName+"</name>\n")
            # write figure
            fo.write("\t\t<bndbox>\n")
            fo.write("\t\t\t<xmin>" + str(int(round(x1))) + "</xmin>\n")
            fo.write("\t\t\t<ymin>" + str(int(round(y1))) + "</ymin>\n")
            fo.write("\t\t\t<xmax>" + str(int(round(x2))) + "</xmax>\n")
            fo.write("\t\t\t<ymax>" + str(int(round(y2))) + "</ymax>\n")
            fo.write("\t\t</bndbox>\n")    
            fo.write("\t</PDFinfo>\n")

        # write out regionless boxes too
        for fp in rawBoxThisPage:
            x1 = fp['boundary']['x1'] 
            y1 = fp['boundary']['y1']
            x2 = fp['boundary']['x2'] 
            y2 = fp['boundary']['y2']
            # transform from PDF to scanned axis
            #y1 = int(round(y1/fracy)); y2 = int(round(y2/fracy))
            #x1 = int(round(x1/fracy-xc)); x2=int(round(x2/fracy-xc))
            y1 = int(round((y1/fracy)*fracyYOLO)); y2 = int(round((y2/fracy)*fracyYOLO))
            x1 = int(round((x1/fracy-xc)*fracxYOLO)); x2=int(round((x2/fracy-xc)*fracxYOLO))
            fo.write("\t<PDFinfoRAW>\n")
            fo.write("\t\t<name>"+'raw'+"</name>\n")
            # write box
            fo.write("\t\t<bndbox>\n")
            fo.write("\t\t\t<xmin>" + str(int(round(x1))) + "</xmin>\n")
            fo.write("\t\t\t<ymin>" + str(int(round(y1))) + "</ymin>\n")
            fo.write("\t\t\t<xmax>" + str(int(round(x2))) + "</xmax>\n")
            fo.write("\t\t\t<ymax>" + str(int(round(y2))) + "</ymax>\n")
            fo.write("\t\t</bndbox>\n")    
            fo.write("\t</PDFinfoRAW>\n")

        
    # on occation, depending on who is annotating and what labels you have in there
    # a "sub fig caption" can be tagged -- if you don't want this you need to do a little 
    # dance to change a lonely sub-fig-caption into a caption by default. This is hidden in 
    # this function:
    objNames, objSquares = get_annotation_name(d,scount,sfcount,ccount)
    # (other names, except ignored labels, are unchanged)
    
    if plot_diagnostics:
        imgPlot = np.array(Image.open(config.images_jpeg_dir+ d['filename'].values[0]+'.jpeg').convert('RGB'))
    for n,bs in zip(objNames, objSquares):
        # just double check for table captions here
        if 'table' in n and 'caption' in n: import sys; sys.exit()
        if n == '': continue # nothing there, don't put it in!
        fo.write("\t<object>\n")
        fo.write("\t\t<name>"+n+"</name>\n") 
        fo.write("\t\t<bndbox>\n")

        # shrink caption around OCR bounding boxes
        #b = true_box_caption_mod(bs,rotation,bboxes_words, 
        #                        true_overlap = 'area', area_overlap=0.75) 
        b = true_box_caption_mod(bs,rotation,bboxes_words, 
                                true_overlap = 'center') 
        
        xmin = max([b[0]*1.0/d['w'].values[0]*config.IMAGE_W,0]) # have to rescale to output image size
        xmax = min([(b[0]+b[2])*1.0/d['w'].values[0]*config.IMAGE_W,config.IMAGE_W])
        ymin = max([b[1]*1.0/d['h'].values[0]*config.IMAGE_H,0])
        ymax = min([(b[1]+b[3])*1.0/d['h'].values[0]*config.IMAGE_H,config.IMAGE_H])
        fo.write("\t\t\t<xmin>" + str(int(round(xmin))) + "</xmin>\n")
        fo.write("\t\t\t<ymin>" + str(int(round(ymin))) + "</ymin>\n")
        fo.write("\t\t\t<xmax>" + str(int(round(xmax))) + "</xmax>\n")
        fo.write("\t\t\t<ymax>" + str(int(round(ymax))) + "</ymax>\n")
        fo.write("\t\t</bndbox>\n")    
        fo.write("\t</object>\n") 
        if plot_diagnostics:
            # plot all blocks on the bottom
            for bw in bboxes_words:
                xmin, ymin, xmax, ymax = bw
                cv.rectangle(imgPlot, (round(xmin), round(ymin)), 
                             (round(xmax),round(ymax)), (0,125,255),1)   # cyan
            # orig boxes w/o modification
            xmin = max([bs[0],0]) # have to rescale to output image size
            xmax = min([bs[0]+bs[2],d['w'].values[0]])
            ymin = max([bs[1],0])
            ymax = min([(bs[1]+bs[3]),d['h'].values[0]])

            if n == 'figure': 
                col_orig = colfig_orig; col_mod = colfig_mod
            elif n == 'figure caption':
                col_orig = colcap_orig; col_mod = colcap_mod
            elif n == 'table':
                col_orig = coltab; col_mod = coltab
            elif n == 'math formula':
                col_orig = colmath; col_mod = colmath
            else: # a weirdo
                col_orig = (100,100,100); col_mod = (100,100,100)
            cv.rectangle(imgPlot, (round(xmin), round(ymin)), 
                         (round(xmax),round(ymax)), col_orig, 7)   # orig
            # modified boxes
            xmin = max([b[0],0]) # have to rescale to output image size
            xmax = min([b[0]+b[2],d['w'].values[0]])
            ymin = max([b[1],0])
            ymax = min([(b[1]+b[3]),d['h'].values[0]])
            cv.rectangle(imgPlot, (round(xmin), round(ymin)), 
                         (round(xmax),round(ymax)), col_mod, 4)   # mod
        #if '1913ApJ____38__496B_p4' in d['filename'].values[0]:
        #    import sys; sys.exit()

 


    fo.write("</annotation>\n")
    fo.close() 
    if plot_diagnostics:
        Image.fromarray(imgPlot).save(diagnostics_file + d['filename'].values[0] + '.png')
        del imgPlot      
    #if iw>5: import sys; sys.exit()

        
        
        
if yt.is_root():
    print('Done with generation...')
    if config.check_nans:
        print('... also checking nans ...')
    if config.check_parse:
        print('... also checking parsability of annotations ...')

    annotations = []
    for ns, vals in sorted(my_storage.items()):
        if vals is not None:
            annotations.append(vals[0])
        
    X_full = np.array(annotations)

    # get all labels
    LABELS = [] # collect all labels
    for ann in X_full:
        tree = ET.parse(ann)
        for elem in tree.iter(): 
            if 'filename' in elem.tag:
                fname = elem.text
            if 'object' in elem.tag:
                for attr in list(elem):
                    if 'name' in attr.tag:
                        if attr.text is not None:
                            LABELS.append(attr.text)
    # endevor to parse the full annotation
    LABELS = np.unique(LABELS).tolist()    
    
    # copy all x-full
    shapes = []; 
    for ann in X_full:
        tree = ET.parse(ann)
        for elem in tree.iter(): 
            if 'filename' in elem.tag:
                fname = elem.text
                #check NaN?
                if config.check_nans:
                    noFile = True
                    try:
                        image_np = np.load(config.save_binary_dir+'binaries/'+ fname.split('/')[-1])['arr_0']
                    except:
                        noFile = False
                    if noFile:
                        image_np = image_np.astype(np.float32) / 255.0
                        if np.any(np.isnan(image_np)):
                            print('NaN found in', fname.split('/')[-1])
                        try:
                            shapes.append(image_np.shape[2])
                        except:
                            shapes.append(-1)
        # endevor to parse the full annotation
        if config.check_parse:
            check_for_file = True
            if not generate_features: check_for_file = False
            try:
                iname, tb = parse_annotation([ann],LABELS,
                                             debug=True,
                                             check_for_file=check_for_file)
            except:
                print('trouble parsing for:', ann)
    # check for same shapes
    if len(np.unique(shapes)) > 1:
        print('to many 3rd shapes:',np.unique(shapes))        
