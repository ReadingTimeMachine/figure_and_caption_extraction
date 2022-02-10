# # for benchmarks, otherwise set to None
# # NON-commercial PubLayNet
# ocr_results_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/OCR_processing_pmcnoncom/'
# nRandom_ocr_image = 1500 # for testing
# full_article_pdfs_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/data/PMC_noncom/pdfs/'
# images_jpeg_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Pages_pmcnoncom/RandomSingleFromPDFIndexed/'
# tmp_storage_dir = None

# scanbank
ocr_results_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/OCR_processing_scanbank/'
nRandom_ocr_image = 1500 # for testing
full_article_pdfs_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/data/scanbank/etds/'
images_jpeg_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Pages_scanbank/RandomSingleFromPDFIndexed/'
tmp_storage_dir = None

# # back to config-file defaults
# ocr_results_dir = None
# nRandom_ocr_image = None #3 # for testing
# full_article_pdfs_dir = None
# images_jpeg_dir = None
# tmp_storage_dir = None

import config

debug = False

# YOU NEED TO DO download_ocr_and_pdfminer_whole_pages_batch.py BEFORE THIS ONE

# naming structure here
nprocs = config.nProcs
if nprocs > 1:
    inParallel = True
images_dir = config.images_jpeg_dir

# print every nth output
mod_output = 10

# OCR params
#config="-l eng --oem 1 --psm 12"
configOCR="--oem 1 --psm 12"


pdffigures_dpi = 300 # this makes around the right size
# NOTE: processing will be done at 2x this and downsized for OCR


# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
    
import numpy as np
import yt
import time  

from wand.image import Image as WandImage
from wand.color import Color
from PIL import Image
from os import remove
import pickle

import pytesseract

if inParallel:
    yt.enable_parallelism()
    
from ocr_and_image_processing_utils import get_already_ocr_processed, find_pickle_file_name, \
   get_random_page_list, find_squares_auto, cull_squares_and_dewarp, angles_results_from_ocr




# -----------------------------------------------

# Get all already done pages... 
wsAlreadyDone = get_already_ocr_processed(ocr_results_dir=ocr_results_dir)

# find the pickle file we will process next
pickle_file_name = find_pickle_file_name(ocr_results_dir=ocr_results_dir)

if yt.is_root(): print('working with pickle file:', pickle_file_name)

# get randomly selected articles and pages
if config.ocr_list_file is None:
    ws, pageNums, pdfarts = get_random_page_list(wsAlreadyDone,
                                                 full_article_pdfs_dir=full_article_pdfs_dir,
                                                nRandom_ocr_image=nRandom_ocr_image)
else:
    if yt.is_root(): print('Using a OCR list -- ', config.ocr_list_file)
    import pandas as pd
    df = pd.read_csv(config.ocr_list_file)
    ws = df['filename'].values
    pageNums = df['pageNum'].values
    pdfarts = ws.copy()
    if yt.is_root(): print('have ', len(ws), ' entries')
    

wsInds = np.arange(0,len(ws))
    
# debug
#wsInds = wsInds[:6]
#wsInds = wsInds[90:]

if images_jpeg_dir is None: images_jpeg_dir = config.images_jpeg_dir
if tmp_storage_dir is None: tmp_storage_dir = config.tmp_storage_dir


my_storage = {}

# count:
start_time = time.time()
if yt.is_root(): print('START: ', time.ctime(start_time))
times_tracking = np.array([]) # I don't think this is used anymore...


for sto, iw in yt.parallel_objects(wsInds, nprocs, storage=my_storage):    
    sto.result_id = iw # id for parallel storage

    ######################## GET PDF AND MAKE IMAGE ###################

    # read PDF file into memory, if using PDFs
    if pdfarts is not None:
        wimgPDF = WandImage(filename=ws[iw] +'[' + str(int(pageNums[iw])) + ']', 
                            resolution=pdffigures_dpi*2, format='pdf') #2x DPI which shrinks later
        thisSeq = wimgPDF.sequence
    else: # bitmaps, formatting for whatfollows
        thisSeq = [ws[iw]]
        
    imPDF = thisSeq[0] # load PDF page into memory
    iimPDF = pageNums[iw] # get page number as well

    if pdfarts is not None: # have PDFs
        checkws = ws[iw].split('/')[-1].split('.pdf')[0] # for outputting file
        
        # make sure we do our best to capture accurate text/images
        imPDF.resize(width=int(0.5*imPDF.width),height=int(0.5*imPDF.height))
        imPDF.background_color = Color("white")
        imPDF.alpha_channel = 'remove'
        WandImage(imPDF).save(filename=images_jpeg_dir + checkws + '_p'+str(iimPDF) + '.jpeg')
        del imPDF
        
        # also for tempTiff -- have to redo here
        wimgPDF = WandImage(filename=ws[iw] +'[' + str(int(pageNums[iw])) + ']', 
                            resolution=pdffigures_dpi*2, format='pdf') #2x DPI which shrinks later
        thisSeq = wimgPDF.sequence
        imPDF = thisSeq[0]
        imPDF.resize(width=int(0.5*imPDF.width),height=int(0.5*imPDF.height))
        imPDF.background_color = Color("white")
        imPDF.alpha_channel = 'remove'
        
        # save a temp TIFF file for OCR
        tmpFile = tmp_storage_dir + checkws + '_p'+str(iimPDF) + '.tiff'
        WandImage(imPDF).save(filename=tmpFile)
        del imPDF
        
        imOCRName = tmpFile
        imgImageProc = images_jpeg_dir + checkws + '_p'+str(iimPDF) + '.jpeg'
    else: # bitmaps or jpegs -- no tiffs!
        if 'bmp' in ws[iw]:
            checkws = ws[iw].split('/')[-1].split('.bmp')[0] # for outputting file
        elif 'jpeg' in ws[iw]:
            checkws = ws[iw].split('/')[-1].split('.jpeg')[0] # for outputting file  
        else:
            checkws = ws[iw].split('/')[-1].split('.jpg')[0] # for outputting file  
        # read in and copy to jpeg 
        im = Image.open(ws[iw])
        im.save(images_jpeg_dir + checkws + '_p'+str(iimPDF) + '.jpeg', quality=95)
        imOCRName = images_jpeg_dir + checkws + '_p'+str(iimPDF) + '.jpeg'
        imgImageProc = images_jpeg_dir + checkws + '_p'+str(iimPDF) + '.jpeg'
        del im

    if debug: print('reading in image:', checkws + '_p'+str(iimPDF) + '.jpeg')
    
    # OCR the tiff image
    # use hocr to grab rotations of text
    with Image.open(imOCRName).convert('RGB') as textImg:
        if debug: print('starting OCR...')
        hocr = pytesseract.image_to_pdf_or_hocr(textImg,  config="--oem 1 --psm 12", extension='hocr')
        if debug: print('done finding OCR')
        
    # get some info useful for squarefinding (by taking out text blocks)
    # these will be bounding boxes of words, rotations of text
    results_def, rotations = angles_results_from_ocr(hocr)
    
    # onto square finding
    # open img in grayscale 
    with Image.open(imgImageProc) as img:
        saved_squares, color_bars = find_squares_auto(img, results_def, rotations) # 4 ways
        saved_squares_culled, cin1,cout1 = cull_squares_and_dewarp(np.array(img.convert('RGB')), 
                                                                   saved_squares.copy())
        # note: cin1 and cout1 are not used as of yet, but could be used to dewarp images

    sto.result = [imgImageProc,ws[iw], hocr, results_def, rotations, saved_squares_culled, color_bars,cin1,cout1]    
    # remove tmp TIFF image for storage reasons if doing PDF processing
    if pdfarts is not None: # have PDFs
        try:
            remove(imOCRName)
        except:
            print('no TIFF file to remove for:', imOCRName)

    if iw%mod_output == 0: print('On ' + str(iw) + ' of ' + str(len(wsInds))+ ' at ' + str(time.ctime(time.time())))
    
    
    
    
if yt.is_root():
    full_run_squares = []; full_run_ocr = []; full_run_rotations = []; wsout = []
    full_run_hocr = []; centers_in = []; centers_out = []; color_bars = []
    full_run_pdf = []
  
    for ns, v in sorted(my_storage.items()):
        if v is not None:
            wsout.append(v[0])
            full_run_pdf.append(v[1])
            full_run_hocr.append(v[2])
            full_run_ocr.append(v[3])
            full_run_rotations.append(v[4])
            full_run_squares.append(v[5])
            color_bars.append(v[6])
            centers_in.append(v[7])
            centers_out.append(v[8])

        
    # do a little test save here - locations of squares and figure caption boxes
    with open(pickle_file_name, 'wb') as ff:
        pickle.dump([wsout, full_run_squares, full_run_ocr, full_run_rotations, 
                     full_run_pdf, full_run_hocr, color_bars,
                      centers_in, centers_out], ff)
        
    print("DONE at", time.ctime(time.time()))

        
