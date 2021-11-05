import config
from glob import glob
import os
import pandas as pd
from yt import is_root
from lxml import etree
import pickle
import numpy as np


def get_all_ocr_files():
    ocrFiles = []
    ocrFilesAll = [config.ocr_results_dir + config.pickle_file_head + '*.pickle']
    for f in ocrFilesAll:
        if '*' not in f:
            ocrFiles.append(f)
        else:
            fs = glob(f)
            for ff in fs:
                ocrFiles.append(ff)
    return ocrFiles


def make_ann_directories():
    fileStorage = config.save_binary_dir
    IMAGE_H = config.IMAGE_H; IMAGE_W = config.IMAGE_W

    # where shall we store all things?
    imgDirAnn = fileStorage + config.ann_name + str(int(IMAGE_H)) + 'x' + str(int(IMAGE_W))  + '_ann/'
    # this one would have been made in parallel_run_pdffigures2.py
    imgDirPDF = fileStorage + config.ann_name + str(int(IMAGE_H)) + 'x' + str(int(IMAGE_W))  + '_pdffigures2/'

    # check these all exist
    if not os.path.exists(fileStorage):
        os.mkdir(fileStorage)
    if not os.path.exists(fileStorage+'binaries/'):
        os.mkdir(fileStorage+'binaries/')

    # get bad skews
    if config.bad_skews_file is not None:
        badskews = pd.read_csv(config.make_sense_dir+config.bad_skews_file, delimiter='(')
        badskewsList = badskews.index.values.tolist()
        if is_root(): print('--- using a bad skew/bad annotations file ---')
    else:
        badskewsList = [-1]
        
    return imgDirAnn, imgDirPDF, badskewsList


def collect_ocr_process_results(ocrFiles, debug = True):
    # we'll need this to grab only the text
    paragraphs = []; ws = []; squares = []; paragraphs_unskewed = []; pdfwords = []; html = []
    rotations = []
    # loop and grab
    for cp in ocrFiles:
        with open(cp, 'rb') as f:
            wsout, full_run_squares, full_run_ocr, full_run_rotations, \
                 full_run_pdf, full_run_hocr, color_bars,\
                  centers_in, centers_out = pickle.load(f)  

        # here -- generate HTML text, and paragraphs
        full_run_htmlText = []; full_run_paragraphs = []
        for ihocr,hocr in enumerate(full_run_hocr):
            if debug: 
                if ihocr%200 == 0 : print('on', ihocr,'of',len(full_run_hocr))
            # translate to text to find namespace for xpath
            htmlText = hocr.decode('utf-8')
            full_run_htmlText.append(htmlText)
            # grab namespace
            nameSpace = ''
            for l in htmlText.split('\n'):
                if 'xmlns' in l:
                    nameSpace = l.split('xmlns="')[1].split('"')[0]
                    break
            tree = etree.fromstring(hocr)
            ns = {'tei': nameSpace}

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

            full_run_paragraphs.append(ocr_par)
            
        # splits
        for i,w in enumerate(wsout):
            wsout[i] = w.split('/')[-1]

        ws.extend(wsout); paragraphs.extend(full_run_paragraphs); squares.extend(full_run_squares);
        html.extend(full_run_htmlText); 
        rotations.extend(full_run_rotations)
        
    return ws, paragraphs, squares, html, rotations




def get_makesense_info_and_years():
    msf = glob(config.make_sense_dir + 'labels_*csv')
    mysquares = []; myfnames = []; myws=[]; myhs=[] 
    for f in msf:
        d = pd.read_csv(f, names=['class','x','y', 'w','h','fname','wm','hm'])
        # collapse onto unique fig names
        fns = d['fname'].unique()
        for ff in fns:
            mys = []#; mycs = []
            d2 = d.loc[d['fname']==ff]
            ff2 = ff[:ff.rfind('.')+1]
            if ff2[-1] == '.': ff2 = ff2[:-1]
            myfnames.append(ff2)
            for index, row in d2.iterrows():
                mys.append((row['x'],row['y'],row['w'],row['h'], row['class'].replace('_',' ')))
                w = row['wm']; h = row['hm']
            mysquares.append(mys); myws.append(w); myhs.append(h)
    # double check that these are also ones with features
    myfnames1 = []; mysquares1 = []; myws1 = []; myhs1 = []
    # check that we actually have these... for some reason
    for ww,ss,wsw,hsw in zip(myfnames,mysquares,myws,myhs):
        if ww+'.jpeg' in df.index.values.tolist():
            myfnames1.append(ww); mysquares1.append(ss)
            myws1.append(wsw); myhs1.append(hsw)
        else:
            if is_root(): print('---- for some reason', ww, 'is not in this list -----')
    # all together
    dfMakeSense = pd.DataFrame({'filename':myfnames1, 'squares':mysquares1, 'w':myws1, 'h':myhs1})
    dd = pd.DataFrame({'filename':myfnames1}) # to put in format we had before
    dd = dd.drop_duplicates(subset='filename')
    if is_root(): print('unique =',len(dd), 'pages')
    dd = dd['filename'].values
    msdd = dfMakeSense['filename'].values.tolist()
    msw = dfMakeSense['w'].values; msh = dfMakeSense['h'].values
    dfMakeSense = dfMakeSense.sort_values('filename')
    return dfMakeSense
        
# get all years in these annotations AND the unique years
def get_years(dd):
    years = np.array([]).astype('int') # store years
    years_list = []; pdflist = []
    for ii, ff in enumerate(dd):
        f = '/'+ff+'.' # for weirdness
        pdfbase = ff
        if len(pdfbase) > 0 and pdfbase[:4].isdigit():
            # store years
            #years = np.append(years, int(pdfbase[:4]))
            #years = np.unique(years)
            years_list.append(int(pdfbase[:4]))
            #pdflist.append(dlinks[ind])
            pdflist.append(config.full_article_pdfs_dir + pdfbase.split('_p')[0]+'.pdf')
        else:
            if is_root(): print('no pdf for', f)
            #years = np.append(years, -1)
            #years = np.unique(years)
            years_list.append(-1)

    years_list = np.array(years_list)
    years = np.unique(years_list)
    return years,years_list