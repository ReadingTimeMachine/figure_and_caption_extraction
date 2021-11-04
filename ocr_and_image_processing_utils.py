from glob import glob
import pickle
import config
from yt import is_root
import numpy as np
import cv2 as cv

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
        print('working with:', len(pdfarts), 'full article PDFs, will pull random pages from these')
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
                    
    return ws, pageNums, pdfarts


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
def find_squares_auto(img, deltaBin = 26, deltaBinReplace = 8, areaCutOff = 0.01):    
    
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

    
    # # for each group, get average positions of all squares in that group
    # # grab where we think RA is going to be in our image
    # xLabelRegion = []    
    # # same for dec or y
    # yLabelRegion = []
    # # collect all points in a group
    # imgs = [] # save individual dewarped images

    # note: each bounding box + axis labels will be dewarped and placed individually
    #imout = backtorgb.copy()
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
            
            #x0 = int(round(np.min(centers[:,0] ))); y0 = int(round(np.min(centers[:,1])))
            #x1 = int(round(np.max(centers[:,0]))); y1 = int(round(np.max(centers[:,1])))
            #ssCenters = np.array([ [x0,y0], [x1,y0], [x1,y1], [x0,y1] ])
            #saved_centers.append(ssCenters)

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

                # https://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html
                #M = cv.getPerspectiveTransform(centers_in,centers_out)

                # create a dewarped image
                #ymin = max([0,y-xRange_y])
                #ymax = y+h+xRange_y
                #xmin = max([min([x-yRange_x,x-xRange_x]),0])
                #xmax = x+w+xRange_x
                #imDistorted = imout.copy()[ymin:ymax, xmin:xmax]
                #dst = cv.warpPerspective(backtorgb.copy(),M, (backtorgb.shape[1],backtorgb.shape[0]))
                # save this under images
                #imgs.append(dst)
                #imout[ymin:ymax, xmin:xmax] = dst


            # finally, store average square
            s = np.zeros([4,2],dtype='int32')
            s[0] = [x,y]
            s[1] = [x+w, y]
            s[2] = [x+w, y+h]
            s[3] = [x, y+h]

            saved_squares.append(s)

#                 # find were we think the x-labels will live
#                 xLabel = np.zeros([4,2],dtype='int32')
#                 # all the + in the y-direction is because images are y=0 at top, increse down
#                 # scale labels by size of image area compared to scale?
#                 xmin = max([x-xRange_x,0])
#                 xLabel[0] = [xmin, y+h+xRange_y]
#                 xLabel[1] = [x+w+xRange_x, y+h+xRange_y]
#                 xLabel[2] = [x+w+xRange_x, y+h-xRange_y_up]
#                 xLabel[3] = [xmin, y+h-xRange_y_up]

#                 xLabelRegion.append(xLabel)

#                 # where do we think y-labels will live
#                 yLabel = np.zeros([4,2],dtype='int32')
#                 xmin = max([x-yRange_x,0])
#                 ymin = max([y-yRange_x_up,0])
#                 yLabel[0] = [xmin, ymin]
#                 yLabel[1] = [x+yRange_x_up, ymin]
#                 yLabel[2] = [x+yRange_x_up, y+h+yRange_y]
#                 yLabel[3] = [xmin, y+h+yRange_y]

#                 yLabelRegion.append(yLabel)


    # if noDewarp:
    #     return saved_centers
    # else:
    #     if not return_centers:
    #         return imgs, saved_squares, xLabelRegion, yLabelRegion
    #     else:
    #         return imgs, saved_squares, xLabelRegion, yLabelRegion, centers_in_s, centers_out_s                
    return saved_squares, centers_in_s, centers_out_s