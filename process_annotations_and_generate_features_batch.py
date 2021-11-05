# process MakeSense.ai into annotations and features for mega-yolo training
import config

pdffigures_dpi = 72 # this is the default DPI of coordinates for PDFs for pdffigures2 (docs say 150, but this is a LIE)


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

from annotation_utils import get_all_ocr_files, make_ann_directories, collect_ocr_process_results

# ----------------------------------------------

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

# make all file directories
fileStorage = config.save_binary_dir
imgDirAnn, imgDirPDF, badskewsList = make_ann_directories()

# for saving diagnostics, if you've chosen to do that
diagnostics_file = config.tmp_storage_dir + 'tmpAnnDiags/'

def create_stuff(lock):
    if os.path.isfile(fileStorage + 'done'): os.remove(fileStorage + 'done') # test to make sure we don't move on in parallel too soon
    if yt.is_root():
        #if not os.path.exists(imgDir):
        #    os.makedirs(imgDir)
        # delete all, remake
        #shutil.rmtree(imgDir)
        #os.makedirs(imgDir)
        # delete all, remake
        if not os.path.exists(imgDirAnn):
            os.makedirs(imgDirAnn)
        shutil.rmtree(imgDirAnn)
        os.makedirs(imgDirAnn)
        # if pdf things not there, then make it, but don't delete it if its there
        if not os.path.exists(imgDirPDF):
            os.makedirs(imgDirPDF)
        if config.plot_diagnostics:
            if not os.path.exists(diagnostics_file): # main file
                os.makedirs(diagnostics_file)
            # delete subfiles
            #if not os.path.exists(diagnostics_file + 'orig_ann/'):
            #    os.makedirs(diagnostics_file + 'orig_ann/')
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