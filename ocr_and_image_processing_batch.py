import config

# YOU NEED TO DO download_ocr_and_pdfminer_whole_pages_batch.py BEFORE THIS ONE
# god help me I'm adding tiffs -- but ONLY for annotated pages!!

# use tiff for OCR?  else, jpeg
use_tiff = True 




# # ------------ MAIN RUN ---------------
# # where to store pdf images once processed into jpeg for OCR?
# ###images_pdf_tiff = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Pages/RandomSingleFromPDFTiff/'
# images_pdf = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Pages/RandomSingleFromPDFIndexed/'
# # where to store OCR data after processing?
# pickle_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/OCR_processing/' 
# # what is basename of pickles? (numbers added to greatest)
# pickle_file_head = 'full_ocr_newPDFs_TIFF_take'
# # where are full article PDFs stored?
# full_article_pdfs = '/Users/jillnaiman/tmpADSDownloads/pdfs/'
# # Do we want to check that we won't be grabbing other files, if so, which pickles to check?
# fileCheckArr = ['full_ocr_newPDFs_TIFF_take*pickle'] 
# # ONLY running on annotations:

# params for this file
# how many random images to grab?
#nRandom = config.nRandom_ocr_image #1500
nprocs = config.nProcs
if nprocs > 1:
    inParallel = True
images_dir = config.images_jpeg_dir
#fileCheckArr = [config.ocr_results_dir+config.pickle_file_head + '*.pickle'] # look for already done ones
#pickle_dir = config.ocr_results_dir
#pickle_file_head = config.pickle_file_head
#full_article_pdfs = config.full_article_pdfs_dir

#inParallel = True
#nprocs = 6

# print every nth output
mod_output = 10

# OCR params
#config="-l eng --oem 1 --psm 12"
config="--oem 1 --psm 12"


pdffigures_dpi = 300 # this makes around the right size
# NOTE: processing will be done at 2x this and downsized for OCR


# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
    
#from glob import glob
#import pickle
import numpy as np
import yt
   

if inParallel:
    yt.enable_parallelism()
    
from ocr_and_image_processing_utils import get_already_ocr_processed, find_pickle_file_name, \
   get_random_page_list


# import numpy as np
# import cv2 as cv
# from glob import glob
# import urllib
# import sys
# from PIL import ImageFont, ImageDraw, Image, ImageOps
# # fonts
# fontpath = "../simsun.ttc" # the font that has this "º" symbol

# from IPython.display import display

# import pytesseract                                                                                          
# from pytesseract.pytesseract import TesseractError

# import imutils
# from imutils.object_detection import non_max_suppression
# import argparse
# import yt
# import os
# import matplotlib.pyplot as plt

# sys.path.append('../')
# from utils import grab_random_list

# from scipy.ndimage.interpolation import rotate as rotateImage

# import pickle
# #!conda install -c anaconda lxml
# from lxml import etree

# # fuzzy searches
# import regex
# import re

# from image_grab_utils import find_squares_auto, cull_squares_and_dewarp, generate_subplots, angle_cos, \
#     create_line_groups_y, find_squares_auto_one

# from caption_utils import rotate, Point, doOverlap, overlappingArea

# import time
# import pandas as pd

# #from pdf2image import convert_from_path # this doesn't do great
# from wand.image import Image as WandImage
# from wand.color import Color

# from pdfminer.pdfparser import PDFParser
# from pdfminer.pdfdocument import PDFDocument
# from pdfminer.pdfpage import PDFPage
# from pdfminer.pdfinterp import resolve1

# import yt
# if inParallel:
#     yt.enable_parallelism()
    
# if not (inParallel and yt.enable_parallelism()):
#     nprocs = 1
#     print('reset nprocs = 1')
        
# # -------- for PDF mining -----------------------

# from scipy import stats

# # for generating PDFs
# import requests, bs4
# from urllib.request import Request, urlopen
# import wget

# # -------------------------------------------------

# #pdfdir+pdflink -> for "unpacking" objects with PDFminer
# def flat_iter(obj):
#     yield obj
#     if isinstance(obj, LTContainer):
#         for ob in obj:
#             yield from flat_iter(ob)

# -----------------------------------------------

# Get all already done pages... 
wsAlreadyDone = get_already_ocr_processed()

# find the pickle file we will process next
pickle_file_name = find_pickle_file_name()

if yt.is_root(): print('working with pickle file:', pickle_file_name)

# get randomly selected articles and pages
ws, pageNums = get_random_page_list(wsAlreadyDone)

                
import sys; sys.exit()
  

    

# debug overwrite
#ws = [full_article_pdfs + '1984ApJ___282__345R.pdf'] 
#pageNums = [5] # blank

wsInds = np.arange(0,len(ws))
    
# debug
#wsInds = wsInds[:2]
#wsInds = wsInds[90:]


my_storage = {}

# count:
start_time = time.time()
if yt.is_root(): print('START: ', time.ctime(start_time))
times_tracking = np.array([]) # I don't think this is used anymore...


for sto, iw in yt.parallel_objects(wsInds, nprocs, storage=my_storage):    
    saved_squares_all_pages = []; ws_all_pages = []; results_def_all_pages=[]
    rotations_all_pages=[]; lineNums_all_pages=[];confidences_all_pages=[]
    ocr_par_all_pages=[]; links_all_pages=[];PDFlinkStorage_all_pages=[]
    pageNumStorage_all_pages=[];
    htmlText_all_pages=[];
    centers_in_all_pages = []; centers_out_all_pages = []
    centers_in_1_all_pages = []; centers_out_1_all_pages = []

    sto.result_id = iw # id for parallel storage

    ######################## GET PDF AND MAKE IMAGE ###################

    # save PDF file
    if pdfarts is not None:
        wimgPDF = WandImage(filename=ws[iw] +'[' + str(int(pageNums[iw])) + ']', 
                            resolution=pdffigures_dpi*2, format='pdf') 
        thisSeq = wimgPDF.sequence
    else: # bitmaps
        thisSeq = [ws[iw]]
        
    for iimPDF2, imPDF in enumerate(thisSeq):
        iimPDF = pageNums[iw] # just overrite for single page
        if pdfarts is not None: # have PDFs
            # make sure we do our best to capture accurate text/images
            imPDF.resize(width=int(0.5*imPDF.width),height=int(0.5*imPDF.height))
            imPDF.background_color = Color("white")
            imPDF.alpha_channel = 'remove'

            checkws = ws[iw].split('/')[-1].split('.pdf')[0] # for outputting file
            WandImage(imPDF).save(filename=images_pdf + checkws + '_p'+str(iimPDF) + '.jpeg')
        else: # bitmaps or jpegs
            if 'bmp' in ws[iw]:
                checkws = ws[iw].split('/')[-1].split('.bmp')[0] # for outputting file
            elif 'jpeg' in ws[iw]:
                checkws = ws[iw].split('/')[-1].split('.jpeg')[0] # for outputting file  
            else:
                checkws = ws[iw].split('/')[-1].split('.jpg')[0] # for outputting file  
            # read in and copy to jpeg 
            im = Image.open(ws[iw])
            im.save(images_pdf + checkws + '_p'+str(iimPDF) + '.jpeg', quality=95)
            #im.save(images_pdf + checkws + '_p'+str(iimPDF) + '.tiff', quality=95)
            del im
        
        ws_all_pages.append(images_pdf + checkws + '_p'+str(iimPDF) + '.jpeg')

        ######################### OCR ################################
        
        if debug: print('reading in image:', checkws + '_p'+str(iimPDF) + '.jpeg')

        # (I) Read in image
        fn = images_pdf + checkws + '_p'+str(iimPDF) + '.jpeg' #ws[iw]
        
        #if iw > 0: import sys; sys.exit()

        img = Image.open(fn)

        # im to data
        img = np.array(img)
        img = np.uint8(img) # just in case

        # for later use
        if len(img.shape) < 3:
            backtorgb = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
        else:
            backtorgb = img.copy()
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)


        # UNSKEWED OCR

        # find all the squares on the page -> use this to de-warp
        if debug: print('finding squares...')
        saved_squares = find_squares_auto_one(img.copy())
        if debug: print('done finding squares')

        # final culling of squares & saving of de-warped images
        if debug: print('culling squares....')
        imgs, saved_squares, _, _, cin1,cout1 = cull_squares_and_dewarp(backtorgb.copy(), 
                                                            saved_squares.copy(), 
                                                           return_centers=True)
        if debug: print('done culling squares')
        # OCR the full image, for every de-warped image
        results_def = []
        rotations = [] # keep if rotated text
        lineNums = [] # store line numbers
        angles = []
        confidences = [] # save confidences

        # replace if we don't want to do this based on skewed stuff
        textImg = img.copy()

        # use hocr to grab rotations of text
        if debug: print('starting OCR...')
        hocr = pytesseract.image_to_pdf_or_hocr(textImg,  config="--oem 1 --psm 12", extension='hocr')
        if debug: print('done finding OCR')
        # translate to text to find namespace for xpath
        htmlText = hocr.decode('utf-8')
        # grab namespace
        nameSpace = ''
        for l in htmlText.split('\n'):
            if 'xmlns' in l:
                nameSpace = l.split('xmlns="')[1].split('"')[0]
                break
        tree = etree.fromstring(hocr)
        ns = {'tei': nameSpace}

        # grab words
        words = tree.xpath("//tei:span[@class='ocrx_word']/text()", namespaces=ns)

        # grab bounding boxes too
        bboxesText = tree.xpath("//tei:span[@class='ocrx_word']/@title", namespaces=ns)
        # parse and grab bboxes alone
        bboxes = []
        for b in bboxesText:
            mybb = b.split('bbox ')[1].split(';')[0].split()
            mybb = np.array(mybb).astype('int').tolist()
            bboxes.append(mybb)   
            confidences.append(float(b.split('x_wconf')[1]))

        # now, grab parent structure and see if you can see a "text angle" tag for rotated text
        angles = []; lines = []; ocr_par = []
        for i,angle in enumerate(tree.xpath("//tei:span[@class='ocrx_word']", namespaces=ns)):
            myangle = angle.xpath("../@title", namespaces=ns) # this should be line tag
            par = angle.xpath("../../@title", namespaces=ns) # grab spacing of paragraph blocks for layout stuff later
            par = par[0]
            bb = np.array(par.split(' ')[1:]).astype('int').tolist()
            x = bb[0]; y = bb[1]
            w = bb[2]-x; h = bb[3]-y
            ocr_par.append((x,y,w,h))
            if len(myangle) > 1:
                print('HAVE TOO MANY PARENTS')
            if 'textangle' in myangle[0]:
                # grab text angle
                textangle = float(myangle[0].split('textangle')[1].split(';')[0])
            else:
                textangle = 0.0
            angles.append(textangle)
            # also please grab line number
            myl = angle.xpath("../@id", namespaces=ns) # this should be line tag
            l = myl[0].split("_")
            if int(l[1]) != 1:
                print(' SOMETHING WEIRD HAS happened!!')
                sys.exit()
            lines.append(int(l[2]))

        # put it all together
        for text, bb, rot,l in zip(words,bboxes,angles,lines):
            x = bb[0]; y = bb[1]
            w = bb[2]-x; h = bb[3]-y
            results_def.append( ((x,y,w,h),text) )
            rotations.append(rot)
            lineNums.append(l)

        #####################################################################################

        # now, let's re-run and try to find more squares
        if debug: print('final square finding...')
        #centers_in = []; centers_out = []
        saved_squares = find_squares_auto(img, results_def, rotations) # 4 ways
        _, saved_squares, _, _, centers_in, centers_out = cull_squares_and_dewarp(backtorgb.copy(), 
                                                                                  saved_squares.copy(), 
                                                                                  return_centers=True)
        if debug: print('done with final square finding')

        sys.stdout.flush()
        saved_squares_all_pages.append(saved_squares)
        results_def_all_pages.append(results_def)
        rotations_all_pages.append(rotations)
        lineNums_all_pages.append(lineNums)
        confidences_all_pages.append(confidences)
        ocr_par_all_pages.append(ocr_par)
        PDFlinkStorage_all_pages.append(ws[iw])
        pageNumStorage_all_pages.append(pageNums[iw])
        htmlText_all_pages.append(htmlText)
        centers_in_all_pages.append(centers_in)
        centers_out_all_pages.append(centers_out)
        centers_in_1_all_pages.append(centers_in)
        centers_out_1_all_pages.append(centers_out)


        del results_def
        del rotations
        del lineNums
        del saved_squares
        del confidences
        del ocr_par
        del imgs
        del img
        del hocr
        del tree
        del words
        del htmlText

    # save full things
    sto.result = [ws_all_pages, results_def_all_pages, rotations_all_pages, lineNums_all_pages, 
                  saved_squares_all_pages, confidences_all_pages, ocr_par_all_pages, 
                  #links_all_pages, gifLinkStorage_all_pages,
                  PDFlinkStorage_all_pages,pageNumStorage_all_pages,
                  #downloadLinkStorage_all_pages, 
                  htmlText_all_pages, 
                 centers_in_all_pages, centers_out_all_pages, 
                 centers_in_1_all_pages, centers_out_1_all_pages]#, ocr_par_unskew, words_in_par_unskew]

    if iw%mod_output == 0: print('On ' + str(iw) + ' of ' + str(len(wsInds)))
    
    
if yt.is_root():
    full_run_squares = []; full_run_ocr = []; full_run_rotations = []; full_run_lineNums = []; wsout = []
    full_run_confidences = []; full_run_paragraphs = []; full_run_pdfwords=[]; full_run_pdftextBoxes = []    
    remove_filenames = []
    full_run_links = []; full_run_gifLinkStorage = []; full_run_PDFlinkStorage = []; full_run_pageNumStorage = []; full_run_downloadLinkStorage=[]
    full_run_PDFlayouts = []
    full_run_htmlTextUS = []; full_run_htmlText = []; centers_in = []; centers_out = []
    centers_in_1 = []; centers_out_1 = []
  
    for ns, vals in sorted(my_storage.items()):
        if vals is not None:
            for v in vals[0]:
                wsout.append(v)
            for v in vals[1]:
                full_run_ocr.append(v)
            for v in vals[2]:
                full_run_rotations.append(v)
            for v in vals[3]:
                full_run_lineNums.append(v)
            for v in vals[4]:
                full_run_squares.append(v)
            for v in vals[5]:
                full_run_confidences.append(v)
            for v in vals[6]:
                full_run_paragraphs.append(v)
            #for v in vals[7]:
            #    full_run_links.append(v)
            #for v in vals[8]:
            #    full_run_gifLinkStorage.append(v)
            for v in vals[7]:
                full_run_PDFlinkStorage.append(v)
            for v in vals[8]:
                full_run_pageNumStorage.append(v)
            #for v in vals[11]:
            #    full_run_downloadLinkStorage.append(v)
            for v in vals[9]:
                full_run_htmlText.append(v)
            for v in vals[10]:
                centers_in.append(v)
            for v in vals[11]:
                centers_out.append(v)
            for v in vals[12]:
                centers_in_1.append(v)
            for v in vals[13]:
                centers_out_1.append(v)
        
    # do a little test save here - locations of squares and figure caption boxes
    #pp = pickle_dir + 'full_ocr_results_and_squares' + pickle_file_mod + '.pickle'
    with open(pickle_file_name, 'wb') as ff:
        pickle.dump([wsout, full_run_squares, full_run_ocr, full_run_rotations, 
                     full_run_lineNums, full_run_confidences, full_run_paragraphs, 
                     full_run_links, full_run_gifLinkStorage, full_run_PDFlinkStorage, 
                     full_run_pageNumStorage, full_run_downloadLinkStorage,
                     full_run_htmlText, centers_in, centers_out, 
                     centers_in_1, centers_out_1], ff)
        
    print("DONE at", time.ctime(time.time()))

        
    # remove all PDF files
    #for f in remove_filenames:
    #    os.remove(f)