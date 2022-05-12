from glob import glob
import pickle
import config
from yt import is_root
import numpy as np
import cv2 as cv
from lxml import etree

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import resolve1
import math

def get_already_ocr_processed(ocr_results_dir=None,pickle_file_head=None):
    if ocr_results_dir is None: ocr_results_dir = config.ocr_results_dir
    if pickle_file_head is None: pickle_file_head = config.pickle_file_head
    fileCheckArr = [ocr_results_dir+pickle_file_head + '*.pickle']
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
                fileCheckArr2.append(ff)
    for cp in fileCheckArr2:
        with open(cp, 'rb') as f:
            wsout, _, _, _, _, _, _,_, _ = pickle.load(f)
        # splits
        for i,w in enumerate(wsout):
            wsout[i] = w.split('/')[-1].split('.jpeg')[0]

            wsAlreadyDone.extend(wsout)
    return wsAlreadyDone



# find the current pickle-file name for this run:
def find_pickle_file_name(ocr_results_dir=None, pickle_file_head=None):
    if ocr_results_dir is None: ocr_results_dir = config.ocr_results_dir
    if pickle_file_head is None: pickle_file_head = config.pickle_file_head
    # look for most recent pickle storage file and have "take" number increase by 1:
    pickle_files = glob(ocr_results_dir + pickle_file_head + '*pickle')
    if len(pickle_files) == 0: # new!
        pickle_file_name = ocr_results_dir + pickle_file_head +'1.pickle'
    else:
        nums = []
        for p in pickle_files:
            nums.append(int(p.split('_take')[-1].split('.pickle')[0]))
        newNum = max(nums)+1
        pickle_file_name = ocr_results_dir + pickle_file_head +str(newNum)+'.pickle'
    return pickle_file_name



# find the list of PDFs/Images/Jpegs you want to process
def get_random_page_list(wsAlreadyDone, full_article_pdfs_dir=None,
                         nRandom_ocr_image=None):
    if full_article_pdfs_dir is None: full_article_pdfs_dir=config.full_article_pdfs_dir
    if nRandom_ocr_image is None: nRandom_ocr_image = config.nRandom_ocr_image
    # get list of possible files from what has been downloaded in full article PDFs
    pdfarts = glob(full_article_pdfs_dir+'*pdf')
    if is_root() and len(pdfarts)>0: 
        print('working with:', len(pdfarts), 'full article PDFs, will pull random pages from these')
    elif is_root():
        print('no PDFs, going to look for bit maps')

    # parse and construct random list -- find number of pages in PDF's and select from them randomly too
    if len(pdfarts): # if we have pdfs
        pdfRandInts = np.random.choice(len(pdfarts),len(pdfarts),replace=False)
        # loop and grab random page (if not already processed)
        ws = []; pageNums = []
        iloop = 0
        while (len(ws) < nRandom_ocr_image):
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
        pdfarts = glob(full_article_pdfs_dir+'*bmp')
        # probably
        if len(pdfarts) < nRandom_ocr_image and len(pdfarts) > 0: # have something, but smaller than random
            ws = pdfarts.copy()
            pdfarts = None
            pageNums = np.repeat(0,len(ws))
        elif len(pdfarts) > nRandom_ocr_image:
            print('not random implemented, stopping')
            import sys; sys.exit()
        else:
            if is_root(): print('no bitmaps, looking for jpegs')
            # look for jpegs
            pdfarts = glob(full_article_pdfs_dir+'*jpg')
            if len(pdfarts) == 0:
                pdfarts = glob(full_article_pdfs_dir+'*jpeg')
            if len(pdfarts) == 0:
                print('really NO idea then... stopping')
                import sys; sys.exit()
            else: # found something! carry on!
                if len(pdfarts) < nRandom_ocr_image:
                    ws = pdfarts.copy()
                    pdfarts = None
                    pageNums = np.repeat(0,len(ws))
                else: # gotta grab random ones
                    pageInt = np.random.choice(len(pdfarts),nRandom,replace=False)
                    ws = np.array(pdfarts)[pageInt]
                    pdfarts = None
                    pageNums = np.repeat(0,len(ws))
                    
    return ws, pageNums, pdfarts



# -------------- parsing some results from OCR for later square finding --------

def angles_results_from_ocr(hocr, return_extras=False):
    # if not string yet:
    if type(hocr) != str:
        htmlText = hocr.decode('utf-8')
    else:
        htmlText = hocr#.copy()
        hocr = hocr.encode()
    results_def = []
    rotations = [] # keep if rotated text
    lineNums = [] # store line numbers
    angles = []
    confidences = [] # save confidences

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
    lines_ocr = tree.xpath("//tei:span[@class='ocr_line']/@title", namespaces=ns)

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

    # put it all together -- are lineNums even used??
    for text, bb, rot,l in zip(words,bboxes,angles,lines):
        x = bb[0]; y = bb[1]
        w = bb[2]-x; h = bb[3]-y
        results_def.append( ((x,y,w,h),text) )
        rotations.append(rot)
        lineNums.append(l)

    if not return_extras:
        return results_def, rotations
    else:
        return results_def, rotations, confidences, words, lines_ocr, ocr_par, bboxes
        # words = text sometimes
        # lines_ocr = lines in feature gen
        # ocr_par is bbox_par sometimes, bboxes is bboxes_words sometimes

            

# ---------- square finding with image processing ------------------
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img, deltaBin = 15, max_cos_req=0.01):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []; contours = []
    for gray in cv.split(img):
        for thrs in range(0, 255, deltaBin):
            if thrs == 0:
                binner = cv.Canny(gray, 0, 50, apertureSize=5)
                binner = cv.dilate(binner, None)
            else:
                _retval, binner = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
                del _retval
                if cv.__version__ > '4':
                    contours, _hierarchy = cv.findContours(binner, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                    del _hierarchy
                else:
                    binner, contours, _hierarchy = cv.findContours(binner, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                    del _hierarchy
            #gc.collect() # how important is this here?? MAY want it back...
            if len(contours) == 0:
                if cv.__version__ > '4':
                    contours, _hierarchy = cv.findContours(binner, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                    del _hierarchy; del binner
                else:
                    binner, contours, _hierarchy = cv.findContours(binner, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                    del binner; del _hierarchy
            
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < max_cos_req:
                        squares.append(cnt)
    return squares


# this automatically re-runs find-squares more finely grained if nothing is found
# deltaBin is how many gray-scale bins to slice our image into to start, deltaBinReplace is how small 
#.  it gets if we don't find any squares the first time
def find_squares_auto_one(img, deltaBin = 26, deltaBinReplace = 8, areaCutOff = 0.01):    
    
    # find squares
    squares = find_squares(img, deltaBin = deltaBin)
    
    # preliminary culling
    saved_squares = []; color_bars = []
    for s in squares:
        x,y,w,h = cv.boundingRect(s)

        # check corners for being the size of the page
        if w*h < img.shape[0]*img.shape[1]*0.9:
            saved_squares.append(s)           
            # look for very thin squares -> probably color bars
            # https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
            # bounding box to compute the aspect ratio
            x,y,w,h = cv.boundingRect(s)
            ar = w / float(h)
            # if large w/h or h/w => probably a color bar
            if (ar > 4) or (ar < 0.25):
                color_bars.append(saved_squares.pop())

            # area cut offs for small figs
            area = w*h
            if len(saved_squares) > 0 and area < img.shape[0]*img.shape[1]*areaCutOff:
                saved_squares.pop()

    # if no squares -> try again, smaller delta
    if len(saved_squares) == 0:
        # find squares
        squares = find_squares(img, deltaBin = deltaBinReplace)

        # preliminary culling
        saved_squares = []; color_bars = []
        for s in squares:
            x,y,w,h = cv.boundingRect(s)

            # check corners for being the size of the page
            if w*h < img.shape[0]*img.shape[1]*0.9:
                saved_squares.append(s)           
                # look for very thin squares -> probably color bars
                # https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
                # bounding box to compute the aspect ratio
                x,y,w,h = cv.boundingRect(s)
                ar = w / float(h)
                # if large w/h or h/w => probably a color bar
                if (ar > 4) or (ar < 0.25):
                    color_bars.append(saved_squares.pop())

                # area cut offs for small figs
                area = w*h
                if len(saved_squares) > 0 and area < img.shape[0]*img.shape[1]*areaCutOff:
                    saved_squares.pop()
                    
        # try one more thing -> relaxing max-cos requirement
    if len(saved_squares) == 0:
        # find squares
        squares = find_squares(img, deltaBin = deltaBin, max_cos_req=0.1)

        # preliminary culling
        saved_squares = []; color_bars = []
        for s in squares:
            x,y,w,h = cv.boundingRect(s)

            # check corners for being the size of the page
            if w*h < img.shape[0]*img.shape[1]*0.9:
                saved_squares.append(s)           
                # look for very thin squares -> probably color bars
                # https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
                # bounding box to compute the aspect ratio
                x,y,w,h = cv.boundingRect(s)
                ar = w / float(h)
                # if large w/h or h/w => probably a color bar
                if (ar > 4) or (ar < 0.25):
                    color_bars.append(saved_squares.pop())

                # area cut offs for small figs
                area = w*h
                if len(saved_squares) > 0 and area < img.shape[0]*img.shape[1]*areaCutOff:
                    saved_squares.pop()
                    
    return saved_squares, color_bars

# uses 4 different methods to find squares, culls at end
def find_squares_auto(img, results_culled, angles_culled, 
                      deltaBin = 26, deltaBinReplace = 8, areaCutOff = 0.01, 
                      hog_disk_radius = 10):  
    from skimage import filters, util
    from scipy.stats import mode
    from skimage.morphology import disk
    
    kShmear = (7,5) # smear more along horizontal
    krotShmear = (5,7)    
    
    # im to data
    img = np.array(img)
    img = np.uint8(img) # just in case

    # for later use
    if len(img.shape) < 3:
        backtorgb = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    else:
        backtorgb = img.copy()
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    if len(mode(angles_culled).mode) > 0:
        myRot = mode(angles_culled).mode[0]
    else:
        myRot = -50
    badRotation = False
    if myRot == 90 or myRot == 270:
        rotatedImage = True
        #print('Rotated image detected')
    elif myRot == 0:
        rotatedImage = False
    else:
        print('rotation issue') 
        print(myRot)
        badRotation = True
        

    showImg = backtorgb.copy()
    if not badRotation:
        blocksImg = backtorgb.copy()
        # ------ (1) using Text to help find squares with masking out of text------
        blocksImg[:,:,:] = 255
        # save locations for "fig" tags
        results_fig = []
        for ((startX, startY, w, h), text) in results_culled:
            # fuzzy search
            if (len(text) > 0):    
                # if ('high' not in text.strip().lower()):
                #     if len(text) < 5:
                #         if regex.match( '(FIG){e<=1}', text, re.IGNORECASE ):
                #             results_fig.append( ((startX,startY,w,h), text) )
                #     elif len(text) >= 5 and len(text) < 7:
                #         if regex.match( '(FIGURE){e<=2}', text, re.IGNORECASE ):
                #             results_fig.append( ((startX,startY,w,h), text) )
                # only plot if length of text is > 1!... or 2...
                if len(text) > 1:
                    cv.rectangle( blocksImg,  (startX, startY), (startX+w, startY+h), (0, 0, 255), -1 )

        kuse = kShmear
        # if image is rotated, un-rotate
        if rotatedImage: # have rotated image -> rotate smear
            kuse = krotShmear

        image = blocksImg 
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (7,7), 0)
        thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

        # regular
        kernel = cv.getStructuringElement(cv.MORPH_RECT, kuse)
        dilate = cv.dilate(thresh, kernel, iterations=4)    


        dilateOut = dilate.copy().astype('float64')
        dilateOut = dilateOut.max() - dilateOut # 1.0 = ok, 0.0 = words
        if dilateOut.max() != 0:
            dilateOut /= dilateOut.max()
        else:
            dilateOut *= 0.0
        for k in range(3): 
            sItmp = showImg[:,:,k]
            sItmp[dilateOut == 0] = 255
            showImg[:,:,k] = sItmp

            
    # what is "white" here?
    myThresh = mode(img.flatten()).mode[0]

    # all
    grayShowImg = showImg.copy()
    showImg[showImg < myThresh] = 0    

    # ------ (2) HOG ----------

    # calc HOG
    mag = filters.scharr(img)
    gx = filters.scharr_h(img)
    gy = filters.scharr_v(img)
    _, ang = cv.cartToPolar(gx, gy)

    # ------ find all the squares -------
    saved_squares, c1 = find_squares_auto_one(showImg.copy(), 
                                          deltaBin = deltaBin, 
                                          deltaBinReplace = 
                                          deltaBinReplace, 
                                          areaCutOff = areaCutOff)

    # also for orig image
    saved_squares_orig, c2 = find_squares_auto_one(grayShowImg.copy(), 
                                          deltaBin = deltaBin, 
                                          deltaBinReplace = 
                                          deltaBinReplace, 
                                          areaCutOff = areaCutOff)

    # also for unaltered image
    saved_squares_img, c3 = find_squares_auto_one(backtorgb.copy(), 
                                          deltaBin = deltaBin, 
                                          deltaBinReplace = 
                                          deltaBinReplace, 
                                          areaCutOff = areaCutOff)

    # HOG
    selem = disk(hog_disk_radius)
    #hogShmear = filters.rank.mean(util.img_as_ubyte(mag.copy()/mag.max()), selem=selem)
    hogShmear = filters.rank.mean(util.img_as_ubyte(mag.copy()/mag.max()), footprint=selem)
    saved_squares_hog, c4 = find_squares_auto_one(hogShmear.copy(), 
                                          deltaBin = deltaBin, 
                                          deltaBinReplace = 
                                          deltaBinReplace, 
                                          areaCutOff = areaCutOff)

    saved_squares_final = []; color_bars = c2 # use orig colorbars
    for ss in saved_squares:
        saved_squares_final.append(ss)
    #for cc in c1:
    #    color_bars.append(cc)

    for ss in saved_squares_orig:
        saved_squares_final.append(ss)
    #for cc in c2:
    #    color_bars.append(cc)

    for ss in saved_squares_img:
        saved_squares_final.append(ss)
    #for cc in c3:
    #    color_bars.append(cc)

    for ss in saved_squares_hog:
        saved_squares_final.append(ss)
    #for cc in c4:
    #    color_bars.append(cc)

    return saved_squares_final, color_bars
#    else:
#        return [], []





# from ref[5]
def cluster_points(points, nclusters):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv.kmeans(points, nclusters, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    return centers

# go through and pull out single squares and axis labels
# also return a set of de-warped images around each square
# NOTE: this old one tried to do the whole image
def cull_squares_and_dewarp(backtorgb,saved_squares):

    # NOTE: there has got to be a better way of doing this then re-doing the function each time
    def clockwiseangle_and_distance(point):
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    saved_centers = []; centers_in_s = []; centers_out_s = []

    if len(saved_squares) > 0: # if we have any squares detected
        sgroup = [[saved_squares[0]]]
        x,y,w,h = cv.boundingRect(sgroup[0][0])
        # calculate center
        cx = x+w*0.5; cy = y+h*0.5
        centerGroup = [[cx,cy]]

        for i in range(1,len(saved_squares)):
            x,y,w,h = cv.boundingRect(saved_squares[i])
            cx = x+w*0.5; cy = y+h*0.5
            # check and see if we are close to centers of other sq groups
            flagGroup = False
            for j in range(len(centerGroup)):
                cxg,cyg = centerGroup[j]
                # difference between x & y centers < 5% w/h,in group
                if (np.abs(cx-cxg) < w*0.05) and (np.abs(cy-cyg) < h*0.05):
                    # we are in this group -> add to it
                    sgroup[j].append(saved_squares[i])
                    flagGroup = True

            if not flagGroup: # new group
                sgroup.append([saved_squares[i]]) # xy coords of individual boxes
                centerGroup.append([cx,cy]) # centers of individual boxes

        
        saved_squares = []

        for i in range(len(centerGroup)):
            xs = []; ys = []
            for j in range(len(sgroup[i])):
                for s in sgroup[i][j]:
                    xs.append(s[0]); ys.append(s[1])
            
            # use kmeans to find centers of clusters
            P = np.float32(np.column_stack((xs, ys)))
            nclusters = 4 # assume squares
            if len(P) >= nclusters:
                centers = cluster_points(P,nclusters)
            else:
                centers = P

            # this x,y combo will be used to dewarp
            x = int(round(np.min(centers[:,0] )))
            w = int(round(np.max(centers[:,0])))-x
            y = int(round(np.min(centers[:,1])))
            h = int(round(np.max(centers[:,1])))-y

            # save centers if you want to do dewarping later on
            centers_out = np.float32([[x,y],[x+w,y],[x,y+h],[x+w,y+h]]) # 3 points should *NOT* be colinear

            # sort rotated clockwise
            origin = [x+0.5*w, y+0.5*h]
            refvec = [0,-1]
            centers_in_tmp = sorted(centers,key=clockwiseangle_and_distance)
            centers_out_tmp = sorted(centers_out,key=clockwiseangle_and_distance)

            # to lists
            centers_in = []
            centers_out = []
            for i in range(len(centers_in_tmp)):
                centers_in.append( centers_in_tmp[i].tolist() )
                centers_out.append( centers_out_tmp[i].tolist() )

            # to float precision numpy
            centers_in = np.float32(centers_in)
            centers_out = np.float32(centers_out)

            centers_in_s.append(centers_in); centers_out_s.append(centers_out)

            # finally, store average square
            s = np.zeros([4,2],dtype='int32')
            s[0] = [x,y]
            s[1] = [x+w, y]
            s[2] = [x+w, y+h]
            s[3] = [x, y+h]

            saved_squares.append(s)
               
    return saved_squares, centers_in_s, centers_out_s