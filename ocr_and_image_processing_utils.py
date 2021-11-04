from glob import glob
import pickle
import config
from yt import is_root
import numpy as np

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import resolve1

def get_already_ocr_processed():
    fileCheckArr = [config.ocr_results_dir+config.pickle_file_head + '*.pickle']
    wsAlreadyDone = []
    # loop and grab
    # check for stars:
    fileCheckArr2 = []
    for f in fileCheckArr:
        if '*' not in f:
            fileCheckArr2.append(f)
        else:
            fs = glob(f)
            for ff in fs:
                fileCheckArr2.append(ff.split('/')[-1])
    for cp in fileCheckArr2:
        with open(cp, 'rb') as f:
            wsout, full_run_squares, full_run_ocr, full_run_rotations, \
                full_run_lineNums, full_run_confidences, full_run_paragraphs, \
                full_run_links, full_run_gifLinkStorage, full_run_PDFlinkStorage, \
                full_run_pageNumStorage, full_run_downloadLinkStorage,\
                full_run_htmlText,_,_,_,_ = pickle.load(f)
            # splits
            for i,w in enumerate(wsout):
                wsout[i] = w.split('/')[-1].split('.jpeg')[0]

            wsAlreadyDone.extend(wsout)
    return wsAlreadyDone



# find the current pickle-file name for this run:
def find_pickle_file_name():
    # look for most recent pickle storage file and have "take" number increase by 1:
    pickle_files = glob(config.ocr_results_dir + config.pickle_file_head + '*pickle')
    if len(pickle_files) == 0: # new!
        pickle_file_name = config.ocr_results_dir + config.pickle_file_head +'1.pickle'
    else:
        nums = []
        for p in pickle_files:
            nums.append(int(p.split('_take')[-1].split('.pickle')[0]))
        newNum = max(nums)+1
        pickle_file_name = config.ocr_results_dir + config.pickle_file_head +str(newNum)+'.pickle'
    return pickle_file_name



# find the list of PDFs/Images/Jpegs you want to process
def get_random_page_list(wsAlreadyDone):
    # get list of possible files from what has been downloaded in full article PDFs
    pdfarts = glob(config.full_article_pdfs_dir+'*pdf')
    if is_root() and len(pdfarts)>0: 
        print('working with:', len(pdfarts), 'full article PDFs')
    elif is_root():
        print('no PDFs, going to look for bit maps')

    # parse and construct random list -- find number of pages in PDF's and select from them randomly too
    if len(pdfarts): # if we have pdfs
        pdfRandInts = np.random.choice(len(pdfarts),len(pdfarts),replace=False)
        # loop and grab random page (if not already processed)
        ws = []; pageNums = []
        iloop = 0
        while (len(ws) < config.nRandom_ocr_image):
            if (iloop>=len(pdfarts)*10): # assume ~10 pages per article
                print('no more files!')
                break
            f = pdfarts[pdfRandInts[iloop%len(pdfarts)]]
            # how many pages?
            parsed = True
            with open(f,'rb') as ff:
                try:
                    parser = PDFParser(ff)
                except:
                    print('cant parse parser', f)
                    parsed = False
                if parsed:
                    try:
                        document = PDFDocument(parser)
                    except:
                        print('cant parse at resolve stage', f)
                        parsed = False
                    if parsed:
                        if resolve1(document.catalog['Pages']) is not None:
                            pages_count = resolve1(document.catalog['Pages'])['Count']  
                        else:
                            pages_count = 1
            if parsed:
                # grab a random page
                pageInt = np.random.choice(pages_count,pages_count,replace=False)
                # check for already having
                art = f.split('/')[-1].split('.pdf')[0] + '_p' + str(int(pageInt[0]))
                iloop2 = 0
                while (art in wsAlreadyDone):
                    if (iloop2 >= pages_count): break
                    art = f.split('/')[-1].split('.pdf')[0] + '_p' + str(int(pageInt[iloop2]))
                    iloop2+=1
                # append if found!
                if iloop2-2 < pages_count: # didn't run out of pages
                    ws.append(f); pageNums.append(int(pageInt[iloop2-1]))

            iloop += 1
        if is_root(): print('end loop to get pages of PDFs, iloop=',iloop)
    else: # look for bitmaps or jpegs
        pdfarts = glob(config.full_article_pdfs_dir+'*bmp')
        # probably
        if len(pdfarts) < config.nRandom_ocr_image and len(pdfarts) > 0: # have something, but smaller than random
            ws = pdfarts.copy()
            pdfarts = None
            pageNums = np.repeat(0,len(ws))
        elif len(pdfarts) > config.nRandom_ocr_image:
            print('not random implemented, stopping')
            import sys; sys.exit()
        else:
            if is_root(): print('no bitmaps, looking for jpegs')
            # look for jpegs
            pdfarts = glob(config.full_article_pdfs_dir+'*jpg')
            if len(pdfarts) == 0:
                pdfarts = glob(config.full_article_pdfs_dir+'*jpeg')
            if len(pdfarts) == 0:
                print('really NO idea then... stopping')
                import sys; sys.exit()
            else: # found something! carry on!
                if len(pdfarts) < config.nRandom_ocr_image:
                    ws = pdfarts.copy()
                    pdfarts = None
                    pageNums = np.repeat(0,len(ws))
                else: # gotta grab random ones
                    pageInt = np.random.choice(len(pdfarts),nRandom,replace=False)
                    ws = np.array(pdfarts)[pageInt]
                    pdfarts = None
                    pageNums = np.repeat(0,len(ws))
                    
    return ws, pageNums