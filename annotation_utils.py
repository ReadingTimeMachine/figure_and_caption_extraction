import config
from glob import glob
import os
import pandas as pd
from yt import is_root
from lxml import etree
import pickle
import numpy as np
import json
from pdf2image import convert_from_path
from scipy import stats
import xml.etree.ElementTree as ET


from general_utils import isRectangleOverlap, iou_orig

def get_all_ocr_files(ocr_results_dir=None,pickle_file_head=None):
    if ocr_results_dir is None: ocr_results_dir = config.ocr_results_dir
    if pickle_file_head is None: pickle_file_head = config.pickle_file_head
    ocrFiles = []
    ocrFilesAll = [ocr_results_dir + pickle_file_head + '*.pickle']
    for f in ocrFilesAll:
        if '*' not in f:
            ocrFiles.append(f)
        else:
            fs = glob(f)
            for ff in fs:
                ocrFiles.append(ff)
    return ocrFiles


def make_ann_directories(save_binary_dir=None,make_sense_dir=None, debug=False):
    if save_binary_dir is None: 
        fileStorage = config.save_binary_dir
    else:
        fileStorage = save_binary_dir
    if make_sense_dir is None: make_sense_dir = config.make_sense_dir
    IMAGE_H = config.IMAGE_H; IMAGE_W = config.IMAGE_W

    # where shall we store all things?
    imgDirAnn = fileStorage + config.ann_name + str(int(IMAGE_H)) + 'x' + str(int(IMAGE_W))  + '_ann/'
    # this one would have been made in parallel_run_pdffigures2.py
    imgDirPDF = fileStorage + config.ann_name + str(int(IMAGE_H)) + 'x' + str(int(IMAGE_W))  + '_pdffigures2/'

    imgDir = fileStorage + 'binaries/'

    # get bad skews
    if config.bad_skews_file is not None:
        try:
            #badskews = pd.read_csv(make_sense_dir+config.bad_skews_file,
            #                       delimiter='(')
            badskews = pd.read_csv(make_sense_dir+config.bad_skews_file)

            badskewsList = badskews.index.values.tolist()
            if is_root(): print('--- using a bad skew/bad annotations file ---')
        except:
            if debug: print('no bad ann file found')
            badskewsList = [-1]
    else:
        badskewsList = [-1]
        
    return imgDir, imgDirAnn, imgDirPDF, badskewsList


def collect_ocr_process_results(ocrFiles, debug = True, imod=1000, get_paragraphs=True):
    # we'll need this to grab only the text
    paragraphs = []; ws = []; squares = []; paragraphs_unskewed = []; pdfwords = []; html = []
    rotations = []; colorbars = []
    # loop and grab
    for icp,cp in enumerate(ocrFiles):
        if debug: 
            if is_root(): print('##### OCR retrieval FILE: on', icp+1,'of',len(ocrFiles), ' ##### ')
        with open(cp, 'rb') as f:
            wsout, full_run_squares, full_run_ocr, full_run_rotations, \
                 full_run_pdf, full_run_hocr, color_bars,\
                  centers_in, centers_out = pickle.load(f)  

        if get_paragraphs:
            # here -- generate HTML text, and paragraphs
            full_run_htmlText = []; full_run_paragraphs = []
            for ihocr,hocr in enumerate(full_run_hocr):
                if debug: 
                    if ihocr%imod == 0 and is_root(): print('--- OCR retrieval: on', ihocr,'of',len(full_run_hocr), '---')
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
            colorbars.extend(color_bars)
        
    return ws, paragraphs, squares, html, rotations, colorbars




def get_makesense_info_and_years(df,make_sense_dir=None):
    if make_sense_dir is None: make_sense_dir = config.make_sense_dir
    msf = glob(make_sense_dir + 'labels_*csv')
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

def get_cross_index(d,df,img_resize,images_jpeg_dir=None):
    """
    d - subset dataframe from a makesense data frame
    df - full list of OCR results
    """
    if images_jpeg_dir is None: images_jpeg_dir = config.images_jpeg_dir
    # get image -- with some checks
    baseName = images_jpeg_dir +d['filename'].values[0]
    if os.path.isfile(baseName + '.jpeg'):
        fname = baseName + '.jpeg'
    elif os.path.isfile(baseName + '.jpg'):
        fname = baseName + '.jpg'
    else:
        try:
            fname = glob(baseName + '*')[0]
        except:
            print('no file found! stopping...')
            import sys; sys.exit()
    if config.plot_diagnostics: # want to plot diagnostic files
        # create images to plot upon
        imgDiag = Image.open(fname)
        imgDiagResize = cv.resize(np.array(imgDiag).astype(np.uint8),
                                 img_resize,fx=0, fy=0, 
                                 interpolation = cv.INTER_NEAREST)            
    # get height, width of orig image
    # reshape to this
    fracxDiag = d['w'].values[0]*1.0/img_resize[0]
    fracyDiag = d['h'].values[0]*1.0/img_resize[1]
    # in order to check for file in other list
    indh = fname.split('/')[-1]
    
    # grab OCR
    goOn = True
    try:
        dfsingle = df.loc[indh]#['hocr']
    except:
        goOn = False
        print('no index for :', indh)
        
    return goOn, dfsingle, indh, fracxDiag, fracyDiag, fname


def get_pdffigures_info(jsonfile, page,ff,d,pdffigures_dpi=72, 
                       full_article_pdfs_dir=None):
    if full_article_pdfs_dir is None: full_article_pdfs_dir = config.full_article_pdfs_dir
    IMAGE_W = config.IMAGE_W
    IMAGE_H = config.IMAGE_H
    figsThisPage = []; rawBoxThisPage =[]
    xc = -1; fracy = -1; 
    fracyYOLO, fracxYOLO = -1, -1
    if os.path.isfile(jsonfile):
        # read in pdffigures2 json
        with open(jsonfile,'r') as ff3:
            try:
                fj = json.loads(ff3.read())    
            except:
                print('no json for ', jsonfile)
                fj = {}
                fj['figures'] = []
                fj['regionless-captions'] = []
        # only want objects on our specific page
        figsThisPage = []
        for fb in fj['figures']:
            if fb['page'] == page:
                figsThisPage.append(fb)
        # also track raw boxes
        rawBoxThisPage = []
        for fb in fj['regionless-captions']:
            if fb['page'] == page:
                rawBoxThisPage.append(fb)

        if (len(figsThisPage) > 0) or (len(rawBoxThisPage) > 0):
            imgPDF = convert_from_path(full_article_pdfs_dir+ff.split('_p')[0]+'.pdf', dpi=pdffigures_dpi, 
                                       first_page=page+1,last_page=page+1)[0]
            # size of the pdffigures2 PDF conversion?
            imgPDFsize = imgPDF.size

            imgScanSize = (d['w'].values[0]*1.0,d['h'].values[0]*1.0) # this order for historic purposes

            # also, what is ratio of YOLO image to other?
            fracxYOLO = IMAGE_W*1.0/imgScanSize[0]; fracyYOLO = IMAGE_H*1.0/imgScanSize[1]

            # check ratios -- scanned pages can be subsets of full PDFs
            rDPI = imgPDFsize[0]/imgPDFsize[1]; rImg = imgScanSize[0]/imgScanSize[1]
            if rDPI == rImg:
                xc = 0
            else: # we have a border that has been cut and need to re-size
                dX = rDPI*imgScanSize[1] # dX1/dY1 * dY2
                xc = (dX-imgScanSize[0])*0.5
            # also for PDFs
            rPDF = imgPDFsize[0]/imgPDFsize[1]
            if rPDF == rImg:
                xcpdf = 0
            else:
                dXPDF = rPDF*imgScanSize[1]
                xcpdf = (dXPDF-imgScanSize[0])*0.5

            fracy = imgPDFsize[1]*1.0/imgScanSize[1]

    return figsThisPage, rawBoxThisPage, xc, fracy, fracyYOLO, fracxYOLO


def get_annotation_name(d,scount,sfcount,ccount,ignore_ann_list=None):
    if ignore_ann_list is None: ignore_ann_list=config.ignore_ann_list
    # take out any overlapping subfigs
    #print(d)
    taken_sqs = []; subsq = []
    for s in d['squares'].values:
        for ss in s: # weird formatting
            if ss[-1] != 'sub fig caption':
                taken_sqs.append(ss)
            else:
                subsq.append(ss)
    #print(taken_sqs)
    #print(subsq)
    for ss in subsq: # check how much overlapping with other things
        x1min = ss[0]; y1min = ss[1]; x1max = ss[0]+ss[2]; y1max = ss[1]+ss[3]
        w1,h1 = x1max-x1min,y1max-y1min
        x1,y1 = x1min+0.5*w1, y1min+0.5*h1
        noOverlapTaken = False
        for ts in taken_sqs:
            x2min = ts[0]; y2min = ts[1]; x2max = ts[0]+ts[2]; y2max = ts[1]+ts[3]
            w2,h2 = x2max-x2min,y2max-y2min
            x2,y2 = x2min+0.5*w2, y2min+0.5*h2
            inter, union, iou = iou_orig(x1, y1, w1, h1, x2, y2, w2, h2, 
                                             return_individual = True)
            #print(inter/(w1*h1))
            if (inter/(w1*h1)) > 0.25: # 25% of area or less is overlapping, otherwise, we have a hit
                noOverlapTaken = True
        if not noOverlapTaken:
            taken_sqs.append(ss)

    #for s in d['squares'].values:
    #    for b in s: # no idea... formatting somewhere
    #print('---')
    #print(taken_sqs)
    diagLabs = []
    for ib,b in enumerate(taken_sqs):
        gotSomething = True ; notSubfig = True    
        if 'sub fig' in b[-1]: notSubfig = False
        # look for some things that we are ignoring
        # NOTE: some of this is redundant!!
        if b[-1] == 'no label': return [''],[]
        if b[-1] in ignore_ann_list: #return '' # in ignore list?
            #diagLabs.append('')
            diagLab = ''
            #print(ib,'hi')
        elif notSubfig or ('sub fig' in b[-1] and (scount <= sfcount) and (ccount == 0) and b[-1] != 'no label'): # this is overly convoluted
            diagLab = ''
            if (scount <= sfcount) and (ccount == 0): # call captions sub-figs
                if b[-1] == 'sub fig caption':
                    #print(ib,'hi2')
                    diagLab = 'figure caption'
                else:
                    #print(ib,'hi3')
                    diagLab = b[-1]
            else:
                #print(ib,'hi44')
                diagLab = b[-1]
        #print(diagLab)
        diagLabs.append(diagLab)
    return diagLabs, taken_sqs


def true_box_caption_mod(b,rotation,bboxes_combined, true_overlap=None,
                        area_overlap = 0.75):
    if true_overlap is None: true_overlap=config.true_overlap
    # use only words if requested
    #if bboxes_words is not None: bboxes_combined=bboxes_words 
    x1min = b[0]; y1min = b[1]; x1max = b[0]+b[2]; y1max = b[1]+b[3]
    trueBoxOut = []; borig = [b].copy()[0]
    captionBox = []
    if 'caption' in b[-1].lower():
        indIou = [1e10,1e10,-1e10,-1e10]
        indIou2 = indIou.copy(); #indIou2[0] = 2e10
        indIou2[0] *= 2; indIou2[1] *= 2; indIou2[2] *= 2; indIou2[3] *= 2
        # don't expand super far in y direction, only x
        i10 = indIou[0]; i11=indIou[2]; i20 = indIou2[0]; i21 = indIou2[2]
        if len(rotation) == 0: rotation = [0]
        if stats.mode(rotation).mode[0] != 90:
            i10 = indIou[1]; i11 = indIou[3]; i20 = indIou2[1]; i21 = indIou2[2]
        #icount = 0
        while (i10 != i20) and (i11 != i21):# and icount<1:
            #icount+=1
        #if True:
            indIou2 = indIou
            i10 = indIou[0]; i11=indIou[2]; i20 = indIou2[0]; i21 = indIou2[2]
            if stats.mode(rotation).mode[0] != 90:
                i10 = indIou[1]; i11 = indIou[3]; i20 = indIou2[1]; i21 = indIou2[2]
            for ibb,bb in enumerate(bboxes_combined):
                x2min, y2min, x2max, y2max = bb
                # is within....vs...
                #true_overlap = 'overlap'
                if true_overlap == 'overlap':
                    isOverlapping = isRectangleOverlap((x1min,y1min,x1max,y1max),
                                                       (x2min,y2min,x2max,y2max))
                #center is within
                elif true_overlap == 'center':
                    x2 = 0.5*(x2min+x2max); y2 = 0.5*(y2min+y2max)
                    isOverlapping = (x2 <= x1max) and (x2 >= x1min) and (y2 <= y1max) and (y2 >= y1min)
                # calculate area overlapping with a cut-off
                elif true_overlap == 'area':
                    # calculate area of overlap
                    dx = min([x1max,x2max])-max([x1min,x2min])
                    dy = min([y1max,y2max])-max([y1min,y2min])
                    area = dx*dy
                    if dx<0 and dy<0: area = 0.0
                    isOverlapping=False
                    if area/((x2max-x2min)*(y2max-y2min)) > area_overlap: isOverlapping = True
                    
                # using whichever condition -- change box sizes
                if isOverlapping:
                    xo = indIou[0]; yo = indIou[1]; xo1 = indIou[2]; yo1 = indIou[3]
                    indIou = [ min([xo,bb[0]]), min([yo,bb[1]]), 
                              max([xo1,bb[2]]), max([yo1,bb[3]])]
                    #print(indIou)
                    captionBox.append((bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]))
            if indIou[0] != 1e10: # found 1 that overlapped
                if stats.mode(rotation).mode[0] == 90: # right-side up
                    x1min, x1max = indIou[0],indIou[2]
                else:
                    y1min, y1max = indIou[1],indIou[3]            
        if indIou[0] != 1e10: # found 1 that overlapped
            trueBoxOut.append((indIou[0],indIou[1],indIou[2]-indIou[0],indIou[3]-indIou[1], b[4]))
            #captionBox.append((indIou[0],indIou[1],indIou[2]-indIou[0],indIou[3]-indIou[1]))
        else:
            trueBoxOut.append(b)
    else:
        trueBoxOut.append(b)
        
    return trueBoxOut[0]




