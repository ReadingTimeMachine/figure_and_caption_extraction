import config

import spacy
nlp = spacy.load("en_core_web_sm")

import cv2 as cv
import pandas as pd
from lxml import etree
from PIL import Image
import numpy as np
import pickle


# SPaCY TAGS
# this is from: https://stackoverflow.com/questions/58215855/how-to-get-full-list-of-pos-tag-and-dep-in-spacy
TAG_LIST = np.unique(np.append(['$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC',
        'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
        'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP',
        'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD',
        'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP',
        '``'],[":",".",",","-LRB-","-RRB-","``","\"\"","''",",","$","#","AFX","CC","CD","DT","EX","FW","HYPH","IN","JJ","JJR","JJS","LS","MD","NIL","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB","ADD","NFP","GW","XX","BES","HVS","_SP"])).tolist()
POS_LIST = np.unique(np.append(['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
        'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SPACE', 'SYM', 'VERB',
        'X'],["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"])).tolist()
DEP_LIST = np.unique(np.append(['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod',
        'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp',
        'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det',
        'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod',
        'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp',
        'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct',
        'quantmod', 'relcl', 'xcomp'],["ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl", "intj", "mark", "meta", "neg", "nn", "npmod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp", "nummod"])).tolist()

# options for angles
angles = np.array([0, 90, 180, 270]) #options
steps = round(256./len(angles))

def generate_single_feature(df, feature_list = None, debug=False, 
                            binary_dir = None, feature_invert=None, 
                           mode='L',maxTag = 125, save_type='uint8', 
                           astype='npz',npzcompressed=False):
    """
    df -- the subset dataframe for this page containing OCR data
    feature_list -- optional, will be config.feature_list if set to None
    binary_dir -- optional, will default to config.save_binary_dir + 'binaries/'
    feature_dir -- optional, will default to config.feature_invert
    mode -- grayscale mode to read in image, will default to "L" for luminance, but can use "P" for palletized
    maxTag -- max number of "color" bands for # of letters & numbers in a word & punctuation
    """
    if feature_list is None: feature_list = config.feature_list
    if binary_dir is None: binary_dir = config.save_binary_dir+'binaries/'
    if feature_invert is None: feature_invert = config.feature_invert
    
    background = 255
    if feature_invert: background = 0
    
    # how many features
    #feature_list = ['grayscale','fontsize','carea boxes','paragraph boxes','fraction of numbers in a word','fraction of letters in a word',
    #            'punctuation','x_ascenders','x_decenders','text angles', 'word confidences','Spacy POS','Spacy TAGs','Spacy DEPs']
    n_features = len(feature_list)
    img_resize = (config.IMAGE_H, config.IMAGE_W)

    ifeature = 0 # keep count of where we are
    
    # read in image as grayscale -- use for all features to replace
    img = np.array(Image.open(config.images_jpeg_dir+df.name).convert(mode))
    # invert?
    if feature_invert: img = 255-img
    # interpolate to size
    imgGray = cv.resize(np.array(img).astype(np.uint8),
                        img_resize,fx=0, fy=0, 
                        interpolation = cv.INTER_NEAREST)
    # save this image
    #imgout = np.zeros([imgGray.shape[0],imgGray.shape[1], n_features]) # to save all features
    imgout = np.zeros([img_resize[0],img_resize[1],n_features])
    # all white
    imgout[:,:,:] = background
    # place holder
    imgOrig = []
    
    # ------- ADD other FEATURES --------
    # we'll parse everything and then just use what is requested
    
    ###### LINE CALCS #####
    h = df['hocr']

    fontshere = []; fontsizeshere = [] # fontsize
    decendershere = [] # x_decenders
    ascendershere = [] # x_ascenders
    anglesshere = [] # text angles
    nameSpace = ''
    for l in h.split('\n'):
        if 'xmlns' in l:
            nameSpace = l.split('xmlns="')[1].split('"')[0]
            break
    ns = {'tei': nameSpace}
    tree = etree.fromstring(h.encode())
    lines = tree.xpath("//tei:span[@class='ocr_line']/@title", namespaces=ns)
    fonts = []
    bboxes = []
    confs = []
    for l in lines:
        x = l.split(';') # each entry
        for y in x:
            if 'x_size' in y: # this is the one that is fontsize
                z = y.strip()
                fontsizeshere.append(float(z.split(' ')[-1]))
                fontshere.append(float(z.split(' ')[-1]))
            elif 'bbox' in y:
                z = y.strip()
                arr=y.split()
                b = np.array(arr[1:]).astype('int')
                bboxes.append( b )
            elif 'x_descenders' in y: # this is the one that is fontsize
                z = y.strip()
                decendershere.append(float(z.split(' ')[-1]))
            elif 'x_ascenders' in y: # this is the one that is fontsize
                z = y.strip()
                ascendershere.append(float(z.split(' ')[-1]))    # now, normalize by page
            elif 'textangle' in y: # this is the one that is fontsize
                z = y.strip()
                anglesshere.append(float(z.split(' ')[-1])) 
                
    ###### WORD CALCS #####
    bboxesw = []
    lines = tree.xpath("//tei:span[@class='ocrx_word']/@title", namespaces=ns)
    text = tree.xpath("//tei:span[@class='ocrx_word']/text()", namespaces=ns)

    # percentage of what?
    smax = 0.0; lmax = 0.0; nmax = 0.0#; pmax = 0.0
    for t,l in zip(text,lines):
        punc = 0
        # count thingies
        numbers = sum(c.isdigit() for c in t)
        letters = sum(c.isalpha() for c in t)
        spaces  = sum(c.isspace() for c in t)   
        #punc  = sum(c.isspace() for c in t)   
        if spaces > smax: smax = spaces
        if numbers > nmax: nmax = numbers
        if letters > lmax: lmax = letters
        if (numbers==0) and (letters==0) and (spaces==0):
            #print(t)
            punc = 1
        x = l.split(';') # each entry
        for y in x:
            if 'bbox' in y:
                z = y.strip()
                arr=y.split()
                b = np.array(arr[1:]).astype('int')
                bboxesw.append( (b, numbers, letters, spaces, punc ) ) 
                
    bboxesw_conf = []
    for i,word in enumerate(tree.xpath("//tei:span[@class='ocrx_word']", namespaces=ns)):
        myangle = word.xpath("../@title", namespaces=ns) # this should be line tag
        par = word.xpath("./@title", namespaces=ns)[0]
        bb = np.array(par.split(';')[0].split(' ')[1:]).astype('int').tolist()
        c = int(par.split('x_wconf')[-1])
        t = word.xpath("./text()",namespaces=ns)[0]
        if len(myangle) > 1:
            print('HAVE TOO MANY PARENTS')
        if 'textangle' in myangle[0]:
            # grab text angle
            textangle = float(myangle[0].split('textangle')[1].split(';')[0])
        else:
            textangle = 0.0    
        bboxesw_conf.append((bb,t,c,textangle))  
        
    ##### CAREA CALCS #####
    bbox_carea = []
    lines = tree.xpath("//tei:div[@class='ocr_carea']/@title", namespaces=ns)
    for l in lines:
        x = l.split(' ')
        b = np.array(x[1:]).astype('int')
        #b[1]:b[3], b[0]:b[2]
        area = (b[3]-b[1])*(b[2]-b[0])
        bbox_carea.append((b,area))
        
    #### PARAGRAPH CALCS #####
    bbox_par = []
    lines = tree.xpath("//tei:p[@class='ocr_par']/@title", namespaces=ns)
    langs = tree.xpath("//tei:p[@class='ocr_par']/@lang", namespaces=ns)
    for l,la in zip(lines,langs):
        x = l.split(' ')
        b = np.array(x[1:]).astype('int')
        area = (b[3]-b[1])*(b[2]-b[0])
        bbox_par.append((b,area,la))
        
    ##### Spacy CALCS #####
    docText = ""
    for b,t,c,ang in bboxesw_conf:
        docText += t + ' '
    doc = nlp(docText)
    pos = []; tags = []; deps = []; is_alpha = []; is_stop = []
    for token in doc:
        pos.append(token.pos_)
        tags.append(token.tag_)
        deps.append(token.dep_)
        is_alpha.append(token.is_alpha)
        is_stop.append(token.is_stop)
    # put back into words -- OCR doesn't match Spacy tokens
    stexts = []
    for b,t,c,ang in bboxesw_conf:
        stexts.append(t)    
    wh = []; ph = []; th = []; dh = []
    bbox_spacy = []
    for token,p,t,d,a,s in zip(doc,pos,tags,deps,is_alpha,is_stop):
        wh.append(token.text); ph.append(p); th.append(t); dh.append(d)
        if token.whitespace_ != '': # new word
            # now here we want to *roughly* cut up each word into its different parts
            # and save these as boxes
            # first, find index of word to match box
            indw = stexts.index(''.join(wh))
            bb, w, c, ang = bboxesw_conf[indw]
            if len(ph) > 1: # multi-part spacy word
                if ang == 0 or ang == 180: # not rotated -- split along x
                    fx = []
                    for w1 in wh:
                        fx.append(len(w1)/len(w)) # fraction of word of this tag
                    # split boxes, first one
                    x1 = bb[0]; y1 = bb[1]; h1 = bb[3]-bb[1]
                    w1 = round((bb[2]-bb[0])*fx[0])
                    bbox_spacy.append((x1,y1,w1,h1,
                                       POS_LIST.index(ph[0]),
                                       TAG_LIST.index(th[0]),
                                       DEP_LIST.index(dh[0])))
                    for iif in range(1,len(fx)):
                        x1 += w1 # start and other's end
                        w1 = round((bb[2]-bb[0])*fx[iif])
                        bbox_spacy.append((x1,y1,w1,h1,
                                           POS_LIST.index(ph[iif]),
                                           TAG_LIST.index(th[iif]),
                                           DEP_LIST.index(dh[iif])))
                elif ang == 90 or ang == 270: # rotated
                    #if ang == 180: print(df['ws'].values[iw], w)
                    fy = []
                    for w1 in wh:
                        fy.append(len(w1)/len(w)) # fraction of word of this tag
                    # split boxes, first one
                    x1 = bb[0]; y1 = bb[1]; w1 = bb[2]-bb[0]
                    h1 = round((bb[3]-bb[1])*fy[0])
                    bbox_spacy.append((x1,y1,w1,h1,
                                       POS_LIST.index(ph[0]),
                                       TAG_LIST.index(th[0]),
                                       DEP_LIST.index(dh[0])))
                    for iif in range(1,len(fy)):
                        y1 += h1 # start and other's end
                        h1 = round((bb[3]-bb[1])*fy[iif])
                        bbox_spacy.append((x1,y1,w1,h1,POS_LIST.index(ph[iif]),
                                           TAG_LIST.index(th[iif]),
                                           DEP_LIST.index(dh[iif])))
                else:
                    print('weird rotation')
                    print(ang, w, df.name)
                    #import sys; sys.exit()
            else: # just one word
                bbox_spacy.append((bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1], 
                                    POS_LIST.index(ph[0]),
                                       TAG_LIST.index(th[0]),
                                       DEP_LIST.index(dh[0])))
            wh = []; ph = []; th=[]; dh=[]


#     #-1. connected components 
#     if 'connected components' in feature_list:
#         imgOrig = img.copy()
#         imgOrig[:,:] = 255
#         **HERE**
    
    
    # 1. save gray
    if 'grayscale' in feature_list:
        imgout[:,:,ifeature] = imgGray
        ifeature += 1
    del imgGray
    
    
    # 2. fontsize
    if 'fontsize' in feature_list:
        # rescale -- not 100% sure which one we want to use here -- using unscaled for now
        fontshere = np.array(fontshere)# - med
        # subtract median
        fontshere -= np.median(fontshere)
        # remove outliers
        fontshere[np.abs(fontshere) > 5*np.std(fontshere)] = 0.0
        if len(fontshere) > 1:
            if fontshere.max() != fontshere.min():
                scales_unscaled = (fontshere-fontshere.min())/(fontshere.max()-fontshere.min())
            else:
                scales_unscaled = fontshere.copy()
                scales_unscaled[:] = background
        else:
            scales_unscaled = [np.array(background)]
        # get img and plot
        imgOrig = img.copy()
        imgOrig[:,:] = background
        # fill
        for b,su in zip(bboxes,scales_unscaled):
            if feature_invert:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(255*su))
            else:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(255-255*su))
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature+=1
        
        
    # 3. carea boxes
    if 'carea boxes' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = background
        for b,a in bbox_carea:
            imgOrig[b[1]:b[3], b[0]:b[2]] = 255-background    
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature+=1
    
    
    # 4. paragraph boxes
    if 'paragraph boxes' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = background
        for b,a,l in bbox_par:
            imgOrig[b[1]:b[3], b[0]:b[2]] = 255-background    
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature += 1
    
    # 5. fraction of numbers in a word & 6. fraction of letters in a word & 7. punctuation
    if 'fraction of numbers in a word' in feature_list:
        if nmax == 0: nmax = 1.0
        imgOrig = img.copy()
        imgOrig[:,:] = background
        for b,n,l,s,p in bboxesw: # numbers, letters, spaces, punc
            if feature_invert:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(n/nmax*maxTag+maxTag)) # always be increasing, just change where we start
            else:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(n/nmax*maxTag))
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature+=1
    
    if 'fraction of letters in a word' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = background
        if lmax == 0: lmax = 1.0
        for b,n,l,s,p in bboxesw: # numbers, letters, spaces, punc
            if feature_invert:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(l/lmax*maxTag+maxTag))
            else:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(l/lmax*maxTag))
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature+=1
        
    if 'punctuation' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = background
        for b,n,l,s,p in bboxesw: # numbers, letters, spaces, punc
            if feature_invert:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(p*maxTag+maxTag))
            else:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(p*maxTag))
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature+=1
        

        
        
    # 8. x_decenders
    if 'x_decenders' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = 255
        decendershere = np.array(decendershere)
        #scales_decenders = (decendershere-min_dec)/(max_dec-min_dec)
        #min_dec = decendershere.min(); max_dec = decendershere.max(); 
        med_dec = np.median(decendershere)
        #if min_dec == max_dec: min_dec = 0; max_dec = 1
        #scales_decenders = (decendershere-min_dec)/(max_dec-min_dec) # around the median
        scales_decenders = (decendershere-med_dec) # around the median
        min_dec = scales_decenders.min(); max_dec = scales_decenders.max()
        if min_dec == max_dec: min_dec = 0; max_dec = 1
        scales_decenders = (scales_decenders-min_dec)/(max_dec-min_dec) # reshift to start at 0
        scales_decenders[scales_decenders<0] = 0
        scales_decenders[scales_decenders>1] = 1
        for b,s in zip(bboxes,scales_decenders):
            #imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(255-s*255))    
            imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(s*255)) # x_ascenders should always be low colors   
        if feature_invert: imgOrig = 255-imgOrig
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature += 1
        
    # 9. x_ascenders
    if 'x_ascenders' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = 255
        ascendershere = np.array(ascendershere)
        # min_asc = ascendershere.min(); max_asc = ascendershere.max()
        # if min_asc == max_asc: min_asc = 0; max_asc = 1
        # scales_ascenders = (ascendershere-min_asc)/(max_asc-min_asc)
        ascendershere += np.median(ascendershere)
        min_asc = ascendershere.min(); max_asc = ascendershere.max()
        scales_ascenders = (ascendershere-min_asc)/(max_asc-min_asc)
        scales_ascenders[scales_ascenders<0] = 0
        scales_ascenders[scales_ascenders>1] = 1
        for b,s in zip(bboxes,scales_ascenders):
            #imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(255-s*255))    
            imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(s*255))    
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        if feature_invert: imgOrig = 255-imgOrig
        ifeature += 1
        
    # 10. text angles
    if 'text angles' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = background
        for b,t,c,ang in bboxesw_conf:
            inda = np.where(ang == angles)[0][0]
            # angle 0 = "normal"
            imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(min([inda*steps,255])))
            if not feature_invert: 
                imgOrig[b[1]:b[3], b[0]:b[2]] = 255-imgOrig[b[1]:b[3], b[0]:b[2]]
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature+=1
    
    # 11. confidence levels
    if 'word confidences' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = background
        for b,t,s,a in bboxesw_conf:
            # default is bad-conf closer to background
            if not feature_invert:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(255-255*s/100.))
            else:
                imgOrig[b[1]:b[3], b[0]:b[2]] = int(round(255*s/100.))
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),img_resize,fx=0, fy=0, interpolation = cv.INTER_NEAREST)
        ifeature += 1


    # 12. Spacy POS
    if 'Spacy POS' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = background
        for b in bbox_spacy: # ideally, these should each be a layer, but...
            imgOrig[b[1]:b[1]+b[3], b[0]:b[0]+b[2]] = b[4]*(255//len(POS_LIST))
            if feature_invert: 
                imgOrig[b[1]:b[1]+b[3], b[0]:b[0]+b[2]] = 255-imgOrig[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature += 1
        
    # 13. Spacy TAG
    if 'Spacy TAGs' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = background
        for b in bbox_spacy:
            imgOrig[b[1]:b[1]+b[3], b[0]:b[0]+b[2]] = b[5]*(255//len(TAG_LIST))
            if feature_invert: 
                imgOrig[b[1]:b[1]+b[3], b[0]:b[0]+b[2]] = 255-imgOrig[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature += 1
        
    # 14. Spacy DEP
    if 'Spacy DEPs' in feature_list:
        imgOrig = img.copy()
        imgOrig[:,:] = background
        for b in bbox_spacy:
            imgOrig[b[1]:b[1]+b[3], b[0]:b[0]+b[2]] = b[6]*(255/len(DEP_LIST))
            if feature_invert: 
                imgOrig[b[1]:b[1]+b[3], b[0]:b[0]+b[2]] = 255-imgOrig[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
        imgout[:,:,ifeature] = cv.resize(np.array(imgOrig).astype(np.uint8),
                                         img_resize,fx=0, fy=0, 
                                         interpolation = cv.INTER_NEAREST)
        ifeature += 1
        
    
    fname = df.name.split('/')[-1]
    fname = fname[:fname.rfind('.')]
    
    # change type as needed
    imgout = imgout.astype(save_type)
    
    # binary save
    if 'pickle' not in astype:
        if npzcompressed:
            ender='.npz'
            with open(binary_dir+fname+'.npz', 'wb') as f:
                np.savez_compressed(f, imgout) # 20 M/file for floats
        else:
            ender='.npy'
            with open(binary_dir+fname+'.npy', 'wb') as f:
                np.savez(f, imgout) # 20 M/file for floats
    else:
        ender='.pickle'
        with open(binary_dir+fname+'.pickle', 'wb') as ff:
            pickle.dump([imgout], ff)
            
    del imgout
    del imgOrig
    return binary_dir+fname+ender



