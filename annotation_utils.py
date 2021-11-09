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

    imgDir = fileStorage + 'binaries/'

    # get bad skews
    if config.bad_skews_file is not None:
        badskews = pd.read_csv(config.make_sense_dir+config.bad_skews_file, delimiter='(')
        badskewsList = badskews.index.values.tolist()
        if is_root(): print('--- using a bad skew/bad annotations file ---')
    else:
        badskewsList = [-1]
        
    return imgDir, imgDirAnn, imgDirPDF, badskewsList


def collect_ocr_process_results(ocrFiles, debug = True, imod=1000):
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
                if ihocr%imod == 0 and is_root(): print('--- OCR retrieval: on', ihocr,'of',len(full_run_hocr))
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




def get_makesense_info_and_years(df):
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

def get_cross_index(d,df,img_resize):
    """
    d - subset dataframe from a makesense data frame
    df - full list of OCR results
    """
    # get image -- with some checks
    baseName = config.images_jpeg_dir +d['filename'].values[0]
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


def get_pdffigures_info(jsonfile, page,ff,d,pdffigures_dpi=72):
    IMAGE_W = config.IMAGE_W
    IMAGE_H = config.IMAGE_H
    figsThisPage = []; rawBoxThisPage =[]
    xc = -1; fracy = -1; 
    fracyYOLO, fracxYOLO = -1, -1
    if os.path.isfile(jsonfile):
        # read in pdffigures2 json
        with open(jsonfile,'r') as ff3:
            fj = json.loads(ff3.read())    
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
            imgPDF = convert_from_path(config.full_article_pdfs_dir+ff.split('_p')[0]+'.pdf', dpi=pdffigures_dpi, 
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


def get_annotation_name(d,scount,sfcount,ccount):
    # take out any overlapping subfigs
    taken_sqs = []; subsq = []
    for s in d['squares'].values:
        for ss in s: # weird formatting
            if ss[-1] != 'sub fig caption':
                taken_sqs.append(ss)
            else:
                subsq.append(ss)
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
    diagLabs = []
    for b in taken_sqs:
        gotSomething = True ; notSubfig = True    
        if 'sub fig' in b[-1]: notSubfig = False
        # look for some things that we are ignoring
        # NOTE: some of this is redundant!!
        if b[-1] == 'no label': return [''],[]
        #    notSubfig = False
        #else:
            #if labslabs[LABELS.index(b[-1])] == -1: notSubfig = False
        #if ignore_mathformula and 'math' in b[-1]: notSubfig = False
        if b[-1] in config.ignore_ann_list: #return '' # in ignore list?
            diagLabs.append('')
        if notSubfig or ('sub fig' in b[-1] and (scount <= sfcount) and (ccount == 0) and b[-1] != 'no label'): # this is overly convoluted
            diagLab = ''
            if (scount <= sfcount) and (ccount == 0): # call captions sub-figs
                if b[-1] == 'sub fig caption':
                    diagLab = 'figure caption'
                else:
                    diagLab = b[-1]
            else:
                diagLab = b[-1]
        diagLabs.append(diagLab)
    return diagLabs, taken_sqs


def true_box_caption_mod(b,rotation,bboxes_combined, true_overlap='overlap'):
    x1min = b[0]; y1min = b[1]; x1max = b[0]+b[2]; y1max = b[1]+b[3]
    trueBoxOut = []; borig = [b].copy()[0]
    captionBox = []
    if 'caption' in b[-1].lower():
        indIou = [1e10,1e10,-1e10,-1e10]
        indIou2 = indIou.copy(); #indIou2[0] = 2e10
        indIou2[0] *= 2; indIou2[1] *= 2; indIou2[2] *= 2; indIou2[3] *= 2
        # don't expand super far in y direction, only x
        i10 = indIou[0]; i11=indIou[2]; i20 = indIou2[0]; i21 = indIou2[2]
        if stats.mode(rotation).mode[0] != 90:
            i10 = indIou[1]; i11 = indIou[3]; i20 = indIou2[1]; i21 = indIou2[2]
        while (i10 != i20) and (i11 != i21):
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




# for parsing annotations
def parse_annotation(split_file_list, labels, feature_dir = '',
                     annotation_dir = '',
                     parse_pdf = False, use_only_one_class=False, 
                    IMAGE_W=512, IMAGE_H=512, debug=False):
    '''
    Parse XML files in PASCAL VOC format.
    
    Parameters
    ----------
    - split_file_list : list of files associated with a split (test, train, val)
    - annotation_dir : annotations files directory (leave empty for no changes)
    - feature_dir : images files directory (leave empty for no changes)
    - labels : labels list
    
    Returns
    -------
    - imgs_name : np array of images files path (shape : images count, 1)
    - true_boxes : np array of annotations for each image (shape : image count, max annotation count, 5)
        annotation format : xmin, ymin, xmax, ymax, class
        xmin, ymin, xmax, ymax : image unit (pixel)
        class = label index
    '''
    
    max_annot = 0
    imgs_name = []
    annots = []
    pdfannots = []; pdfrawannots = []
    pdf_max_annot = 0; pdf_raw_max_annot = 0
    
    # Parse file
    for ann in split_file_list:
        #print(ann)
        annot_count = 0
        boxes = []
        pdfboxes = []; pdfrawboxes = []
        pdf_annot_count = 0; pdf_raw_annot_count = 0
        if len(annotation_dir)>0: # replace
            ann = annotation_dir+ann.split('/')[-1]
        tree = ET.parse(ann)
        for elem in tree.iter(): 
            if 'filename' in elem.tag:
                # replace the default storage?
                if len(feature_dir)>0:
                    iname = feature_dir + elem.text.split('/')[-1]
                    if debug: print(iname)
                else:
                    iname = elem.text
                if not os.path.isfile(iname):
                    iname = iname[:iname.rfind('.')]
                    if debug: print(iname)
                    iname = glob(iname+'*')[0]
                imgs_name.append(iname)
                # if '/Users/jillnaiman' not in thisDir: # probably on google
                #     if len(feature_dir)>0:
                #         iname = feature_dir + elem.text.split('/')[-1]
                #     else:
                #         iname = elem.text
                #     if not os.path.isfile(iname):
                #         iname = iname[:iname.rfind('.')]
                #         iname = glob.glob(iname+'*')[0]
                #     imgs_name.append(iname)
                # else:
                #     if len(feature_dir) > 0:
                #         iname = feature_dir + elem.text.split('/')[-1]
                #     else:
                #         iname = elem.text
                #     #print(iname)
                #     if not os.path.isfile(iname):
                #         iname = iname[:iname.rfind('.')]
                #         iname = glob.glob(iname+'*')[0]
                #     imgs_name.append(iname)
                #print(imgs_name[-1])
                #print(iname)
            if 'width' in elem.tag:
                w = int(elem.text)
            if 'height' in elem.tag:
                h = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:                  
                box = np.zeros((5))
                #yesTag = True
                for attr in list(elem):
                    if 'name' in attr.tag:
                        box[4] = labels.index(attr.text) + 1 # 0:label for no bounding box
                        if use_only_one_class: box[4] = 1
                    if 'bndbox' in attr.tag and 'bndboxOrig' not in attr.tag:
                        annot_count += 1
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                box[0] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                box[1] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                box[2] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                box[3] = int(round(float(dim.text)))
                boxes.append(np.asarray(box))
            if parse_pdf: # look for the info
                if 'PDFinfo' in elem.tag and 'RAW' not in elem.tag:                  
                    pdfbox = np.zeros((5)).tolist()
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            try:
                                pdfbox[4] = attr.text #labels.index(attr.text) + 1 # 0:label for no bounding box
                            except:
                                pdfbox[4] = -1 # something not in the list 
                        if 'bndbox' in attr.tag:
                            pdf_annot_count += 1
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    pdfbox[0] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    pdfbox[1] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    pdfbox[2] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    pdfbox[3] = int(round(float(dim.text)))
                    pdfboxes.append(pdfbox)
                if 'PDFinfoRAW' in elem.tag:                  
                    pdfrawbox = np.zeros((5)).tolist()
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            pdfrawbox[4] = 'figure caption' #labels.index('figure caption') + 1 # guess of raw
                        if 'bndbox' in attr.tag:
                            pdf_raw_annot_count += 1
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    pdfrawbox[0] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    pdfrawbox[1] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    pdfrawbox[2] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    pdfrawbox[3] = int(round(float(dim.text)))
                    pdfrawboxes.append(pdfrawbox)
                    
                    
        if w != IMAGE_W or h != IMAGE_H :
            print('Image size error')
            break
            
        annots.append(np.asarray(boxes))
        pdfannots.append(pdfboxes)
        pdfrawannots.append(pdfrawboxes)
        

        if annot_count > max_annot:
            max_annot = annot_count
        if parse_pdf:
            if pdf_annot_count > pdf_max_annot:
                pdf_max_annot = pdf_annot_count
            if pdf_raw_annot_count > pdf_raw_max_annot:
                pdf_raw_max_annot = pdf_raw_annot_count
           
    # Rectify annotations boxes : len -> max_annot
    imgs_name = np.array(imgs_name)
    #someEmpty = False
    #print(annots)
    #print('max annotation', max_annot)
    if max_annot > 0:
        true_boxes = np.zeros((imgs_name.shape[0], max_annot, 5))
        for idx, boxes in enumerate(annots):
            #print(boxes)
            if len(boxes) > 0:
                true_boxes[idx, :boxes.shape[0], :5] = boxes                
    else: #if len(annots) == 0:
        true_boxes = []
        
    #print(true_boxes)

    # Rectify annotations boxes : len -> max_annot
    if parse_pdf:
        if len(pdfboxes) > 0:
            pdf_true_boxes = np.zeros((imgs_name.shape[0], pdf_max_annot, 5)).tolist()
            for idx, boxes in enumerate(pdfannots):
                for m in range(len(boxes)):
                    pdf_true_boxes[idx][m][:5] = boxes[m]
        else:
            pdf_true_boxes = []
    if parse_pdf:
        if len(pdfrawboxes) > 0:
            pdf_raw_true_boxes = np.zeros((imgs_name.shape[0], pdf_raw_max_annot, 5)).tolist()
            for idx, boxes in enumerate(pdfrawannots):
                for m in range(len(boxes)):
                    pdf_raw_true_boxes[idx][m][:5] = boxes[m]
        else:
            pdf_raw_true_boxes = []
    
    if parse_pdf:
        return imgs_name, true_boxes, pdf_true_boxes, pdf_raw_true_boxes
    else:
        return imgs_name, true_boxes