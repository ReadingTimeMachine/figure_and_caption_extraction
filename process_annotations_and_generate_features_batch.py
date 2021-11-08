# process MakeSense.ai into annotations and features for mega-yolo training
import config

pdffigures_dpi = 72 # this is the default DPI of coordinates for PDFs for pdffigures2 (docs say 150, but this is a LIE)
reRun = False # only toggle on if you want to re-run all of pdffigures2 which can take a while

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
import cv2 as cv
import numpy as np

from annotation_utils import get_all_ocr_files, make_ann_directories, collect_ocr_process_results, \
   get_makesense_info_and_years, get_years, get_cross_index

from ocr_and_image_processing_utils import angles_results_from_ocr

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

# get make sense info
dfMakeSense = get_makesense_info_and_years(df)

# get years and years list
years, years_list = get_years(dfMakeSense['filename'].values)

# storage
my_storage = {}
wsInds = np.linspace(0,len(dfMakeSense['filename'].values)-1,len(dfMakeSense['filename'].values)).astype('int')
# debug
#wsInds = wsInds[:5]
mod_output = 100

# lets do this thing...
if yt.is_root(): print('Making annotation files and features...')

for sto, iw in yt.parallel_objects(wsInds, config.nProcs, storage=my_storage):
    if iw%mod_output == 0: print('On ' + str(iw) + ' of ' + str(len(dfMakeSense['filename'].values)))

    sto.result_id = iw
    
    # some flags to see if we have
    ##gotPDF = False; gotPDFraw = False
    
    #gotSomethings = []
    img_resize=(config.IMAGE_H, config.IMAGE_W)
    #icc = 0 # debugging
    #ii = iw; ff = dd[ii] # old code translation
    
    # subset dataframe
    d = dfMakeSense.loc[dfMakeSense['filename']==dfMakeSense['filename'].values[iw]]
    
    # get squares & save
    gotSomething = False
    scount=0;sfcount=0; ccount = 0
    for b in d['squares'].values:
        gotSomething = True
        # count annotations of different counts -- used for changing subfig captions into captions as needed
        if b[-1] == 'figure': scount +=1
        if b[-1] == 'sub fig caption': sfcount +=1
        if b[-1] == 'figure caption': ccount +=1    
    
    # cross index this makesense data frame with the OCR processing results
    # goOn lets us know if we should continue or not
    goOn, hocr, indh, fracxDiag, fracyDiag, fname = get_cross_index(d,df,img_resize)
    if not goOn: continue

    _, rotation, _, _, _, bbox_par, bboxes_words = angles_results_from_ocr(hocr, return_extras=True)
        
    bboxes_combined = []
    for ibb, bp in enumerate(bboxes_words): 
        bboxes_combined.append(bp)
    for ibb, bp in enumerate(bbox_par): # these are also xmin,ymin,xmax,ymax -- found w/OCR, original page size
        bboxes_combined.append(bp)
        
        
        
    # run pdffigures2 on this thing
    fileExpected = imgDirPDF + fname.split('/')[-1].split('_p')[0] + '.json'
    pdfExpected = config.full_article_pdfs_dir + fname.split('/')[-1].split('_p')[0] + '.pdf'
    # do we have any pdf heres?
    if os.path.exists(pdfExpected):
        if not os.path.exists(fileExpected) or reRun:
            #import sys; sys.exit()
            try:
                subprocess.check_call(
                'java'
                ' -jar {pdffigures_jar_path}'
                ' --figure-data-prefix {pdffigures_dir}'
                ' --save-regionless-captions'
                ' {pdf_path}'.format(
                    pdffigures_jar_path=config.pdffigures_jar_path,
                    pdf_path=pdfExpected,
                    pdffigures_dir=imgDirPDF),
                shell=True)  
            except:
                print('----- ERROR: something wrong has occured with pdffigures2 ------')
                print(' on #', ws[iw])
    else:
        print('no PDF for', fname.split('/')[-1])
        
    import sys; sys.exit()
#    if goOn:
        # # paragraphs from OCR
        # bbox_par = []
        # nameSpace = ''
        # for l in hocr.split('\n'):
        #     if 'xmlns' in l:
        #         nameSpace = l.split('xmlns="')[1].split('"')[0]
        #         break
        # ns = {'tei': nameSpace}
        # tree = etree.fromstring(hocr.encode())
        # # get words
        # bboxes_words = []
        # lines = tree.xpath("//tei:span[@class='ocrx_word']/@title", namespaces=ns)
        # text = tree.xpath("//tei:span[@class='ocrx_word']/text()", namespaces=ns)
        # # get words
        # for t,l in zip(text,lines):
        #     x = l.split(';') # each entry
        #     for y in x:
        #         if 'bbox' in y:
        #             z = y.strip()
        #             arr=y.split()
        #             b = np.array(arr[1:]).astype('int')
        #         elif 'x_wconf' in y:# also confidence
        #             c = y.split('x_wconf')[-1].strip()
        #     bboxes_words.append((b,t,int(c))) 
        # # get paragraphs
        # lines = tree.xpath("//tei:p[@class='ocr_par']/@title", namespaces=ns)
        # langs = tree.xpath("//tei:p[@class='ocr_par']/@lang", namespaces=ns)
        # for l,la in zip(lines,langs):
        #     x = l.split(' ')
        #     b = np.array(x[1:]).astype('int')
        #     area = (b[3]-b[1])*(b[2]-b[0])
        #     bbox_par.append((b,area,la))
        # # combine paragraphs & words
        # bboxes_combined = []
        # for ibb, bp in enumerate(bboxes_words): 
        #     bb,texts,confs = bp 
        #     bboxes_combined.append(bb)
        # for ibb, bp in enumerate(bbox_par): # these are also xmin,ymin,xmax,ymax -- found w/OCR, original page size
        #     bb,aa,ll = bp    
        #     bboxes_combined.append(bb)
        # # grab rotation
        # rotation = dfsave.loc[indh]['rotation']
        
        
        
        
        
        
        
        
#     # run pdffigures2 on this thing
#     fileExpected = imgDirPDF + fname.split('/')[-1].split('_p')[0] + '.json'
#     pdfExpected = pdfStorage + fname.split('/')[-1].split('_p')[0] + '.pdf'
#     # do we have any pdf heres?
#     if os.path.exists(pdfExpected):
#         if not os.path.exists(fileExpected) or reRun:
#             #import sys; sys.exit()
#             try:
#                 subprocess.check_call(
#                 'java'
#                 ' -jar {pdffigures_jar_path}'
#                 ' --figure-data-prefix {pdffigures_dir}'
#                 ' --save-regionless-captions'
#                 ' {pdf_path}'.format(
#                     pdffigures_jar_path=pdffigures_jar_path,
#                     pdf_path=pdfExpected,
#                     pdffigures_dir=imgDirPDF),
#                 shell=True)  
#             except:
#                 print('----- ERROR: something wrong has occured------')
#                 print(' on #', ia)
#     else:
#         print('no PDF for', fname.split('/')[-1])
    #import sys; sys.exit()
        

    #### HERE ####
    if not gotSomething:
        print('nothing for ', ff)
    #gotSomethings.append(gotSomething)

    #print('hey')

    if gotSomething:
        # open up annotation folder
        aloc = imgDirAnn + ff + '.xml'
        floc = fileStorage + ff.split('/')[-1]
        #print(floc)
        # npz not png
        #floc = floc[:floc.rfind('.')+1]
        #if floc[-1] == '.': floc = floc[:-1]
        floc += '.npz'
        
        #import sys; sys.exit()

        # test
        if len(floc.split('/')[-1]) != len(aloc.split('/')[-1]):
            print(floc, aloc)
            print('NOPE')
            import sys; sys.exit()
            
        # before doing anything, check for any "notsures"
        if use_makesense:
            goOn = True
            myLabels = []
            for s in d['squares'].values:
                for b in s: # no idea... formatting somewhere
                    if b[-1] == 'NotSure': # not sure then don't tag it!
                        goOn = False
                        print("not sure for:", ff)
                    elif b[-1] == 'no label':
                        print('no label for', ff)
                    else:
                        myLabels.append(labslabs[LABELS.index(b[-1])])

            # check for that we have something... change eventually
            #. for empty cases
            if goOn:
                if len(np.where(np.array(myLabels) != -1)[0]) == 0: 
                    #goOn = False
                    print("no requested classes for:",ff)
                    
            # check if in bad skew list
            if goOn:
                if ff in badskewsList:
                    print('bad skew/bad annotation for:', ff)
                    goOn = False
        
        if goOn: # surely there is a better way
            fo = open(aloc,"w")
            sto.result = [aloc]
            fo.write("<annotation>\n")
            fo.write("\t<filename>"+floc+"</filename>\n")
            fo.write("\t<size>\n")
            fo.write("\t\t<width>" + str(IMAGE_W) + "</width>\n")
            fo.write("\t\t<height>" + str(IMAGE_H) + "</height>\n")
            fo.write("\t\t<depth>" + str(1) + "</depth>\n")
            fo.write("\t</size>\n")

            # add year info
            fo.write("\t<year>"+str(int(years_list[ii]))+"</year>\n")
            if len(years[years>-1]) > 0:
                fo.write("\t<yearMin>"+str(int(np.min(years[years>-1])))+"</yearMin>\n")
                fo.write("\t<yearMax>"+str(int(np.max(years[years>-1])))+"</yearMax>\n")
                fo.write("\t<yearMean>"+str(int(np.mean(years[years>-1])))+"</yearMean>\n")

            # include pdffigures2 info
            # do we have an associated PDF file?
            #if '_p' in f:
            #    ind = wsLinks.index(f[:f.rfind('.')].split('/')[-1].split('_p')[0])
            #jsonbase = dlinks[ind].split('/')[-1].split('.pdf')[0]+'.json'
            
#             pdfbase = ff
#     if len(pdfbase) > 0:
#         # store years
#         years = np.append(years, int(pdfbase[:4]))
#         years = np.unique(years)
#         years_list.append(int(pdfbase[:4]))
#         #pdflist.append(dlinks[ind])
#         pdflist.append(pdfStorage + pdfbase.split('_p')[0]+'.pdf')

            jsonbase = ff.split('/')[-1].split('_p')[0]+'.json'
            #import sys; sys.exit()
            jsonfile = imgDirPDF + jsonbase
            # now, read the appropriate page's info in
            # get it from _p
            doGoOn = True
            try:
                page = int(ff.split('_p')[-1])
            except:
                print('Issue with finding page for:', ff)
                doGoOn = False
            if os.path.isfile(jsonfile) and doGoOn:
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
                    #print('ding!')

                    #imgPDF = convert_from_path(dlinks[ind], dpi=pdffigures_dpi, 
                    #                           first_page=page+1,last_page=page+1)[0]
                    imgPDF = convert_from_path(pdfStorage+ff.split('_p')[0]+'.pdf', dpi=pdffigures_dpi, 
                                               first_page=page+1,last_page=page+1)[0]
                    # size of the pdffigures2 PDF conversion?
                    imgPDFsize = imgPDF.size

                    imgScanSize = (d['w'].values[0]*1.0,d['h'].values[0]*1.0) # this order for historic purposes
                    
                    #import sys; sys.exit()
                    
                    # also, what is ratio of YOLO image to other?
                    fracxYOLO = IMAGE_W*1.0/imgScanSize[0]; fracyYOLO = IMAGE_H*1.0/imgScanSize[1]
                    
                    # check ratios -- scanned pages can be subsets of full PDFs
                    rDPI = imgPDFsize[0]/imgPDFsize[1]; rImg = imgScanSize[0]/imgScanSize[1]
                    if rDPI == rImg:
                        xc = 0
                    else: # we have a border that has been cut and need to re-size
                        dX = rDPI*imgScanSize[1] # dX1/dY1 * dY2
                        #xc = int(round((dX-imgSize[0])*0.5)) # this is the amount of extra pixels on the PDF side vs scanned page
                        xc = (dX-imgScanSize[0])*0.5
                        #xc = 
                    # also for PDFs
                    rPDF = imgPDFsize[0]/imgPDFsize[1]
                    if rPDF == rImg:
                        xcpdf = 0
                    else:
                        dXPDF = rPDF*imgScanSize[1]
                        xcpdf = (dXPDF-imgScanSize[0])*0.5

                    fracy = imgPDFsize[1]*1.0/imgScanSize[1]
                    
                    #if 'f75aa42c-ce9d-4caf-a6f3-4696af323418' in ff:
                    #    import sys; sys.exit()

                    # write out boxes
                    icc += 1
                    for fp in figsThisPage:
                        gotPDF = True
                        x1 = fp['regionBoundary']['x1'] 
                        y1 = fp['regionBoundary']['y1']
                        x2 = fp['regionBoundary']['x2'] 
                        y2 = fp['regionBoundary']['y2']
                        # transform from PDF to scanned axis to YOLO boxsize
                        y1 = int(round((y1/fracy)*fracyYOLO)); y2 = int(round((y2/fracy)*fracyYOLO))
                        x1 = int(round((x1/fracy-xc)*fracxYOLO)); x2=int(round((x2/fracy-xc)*fracxYOLO))
                        #print(xc)
                        fo.write("\t<PDFinfo>\n")
                        if fp['figType'] == 'Figure':
                            fo.write("\t\t<name>"+'figure'+"</name>\n")
                            capName = 'figure caption'
                        elif fp['figType'] == 'Table':
                            fo.write("\t\t<name>"+'table'+"</name>\n")
                            capName = 'table caption'
                        # write figure/table
                        fo.write("\t\t<bndbox>\n")
                        fo.write("\t\t\t<xmin>" + str(int(round(x1))) + "</xmin>\n")
                        fo.write("\t\t\t<ymin>" + str(int(round(y1))) + "</ymin>\n")
                        fo.write("\t\t\t<xmax>" + str(int(round(x2))) + "</xmax>\n")
                        fo.write("\t\t\t<ymax>" + str(int(round(y2))) + "</ymax>\n")
                        fo.write("\t\t</bndbox>\n")    
                        fo.write("\t</PDFinfo>\n")
                        # write caption
                        x1 = fp['captionBoundary']['x1'] 
                        y1 = fp['captionBoundary']['y1']
                        x2 = fp['captionBoundary']['x2'] 
                        y2 = fp['captionBoundary']['y2']
                        # transform from PDF to scanned axis
                        #y1 = int(round(y1/fracy)); y2 = int(round(y2/fracy))
                        #x1 = int(round(x1/fracy-xc)); x2=int(round(x2/fracy-xc))
                        y1 = int(round((y1/fracy)*fracyYOLO)); y2 = int(round((y2/fracy)*fracyYOLO))
                        x1 = int(round((x1/fracy-xc)*fracxYOLO)); x2=int(round((x2/fracy-xc)*fracxYOLO))
                        fo.write("\t<PDFinfo>\n")
                        fo.write("\t\t<name>"+capName+"</name>\n")
                        # write figure
                        fo.write("\t\t<bndbox>\n")
                        fo.write("\t\t\t<xmin>" + str(int(round(x1))) + "</xmin>\n")
                        fo.write("\t\t\t<ymin>" + str(int(round(y1))) + "</ymin>\n")
                        fo.write("\t\t\t<xmax>" + str(int(round(x2))) + "</xmax>\n")
                        fo.write("\t\t\t<ymax>" + str(int(round(y2))) + "</ymax>\n")
                        fo.write("\t\t</bndbox>\n")    
                        fo.write("\t</PDFinfo>\n")

                    # write out regionless boxes too
                    icc += 1
                    for fp in rawBoxThisPage:
                        gotPDFraw = True
                        x1 = fp['boundary']['x1'] 
                        y1 = fp['boundary']['y1']
                        x2 = fp['boundary']['x2'] 
                        y2 = fp['boundary']['y2']
                        # transform from PDF to scanned axis
                        #y1 = int(round(y1/fracy)); y2 = int(round(y2/fracy))
                        #x1 = int(round(x1/fracy-xc)); x2=int(round(x2/fracy-xc))
                        y1 = int(round((y1/fracy)*fracyYOLO)); y2 = int(round((y2/fracy)*fracyYOLO))
                        x1 = int(round((x1/fracy-xc)*fracxYOLO)); x2=int(round((x2/fracy-xc)*fracxYOLO))
                        fo.write("\t<PDFinfoRAW>\n")
                        fo.write("\t\t<name>"+'raw'+"</name>\n")
                        # write box
                        fo.write("\t\t<bndbox>\n")
                        fo.write("\t\t\t<xmin>" + str(int(round(x1))) + "</xmin>\n")
                        fo.write("\t\t\t<ymin>" + str(int(round(y1))) + "</ymin>\n")
                        fo.write("\t\t\t<xmax>" + str(int(round(x2))) + "</xmax>\n")
                        fo.write("\t\t\t<ymax>" + str(int(round(y2))) + "</ymax>\n")
                        fo.write("\t\t</bndbox>\n")    
                        fo.write("\t</PDFinfoRAW>\n")


            if use_makesense:
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
                if True: # bad formatting, fix, I know
                    for b in taken_sqs:
                        gotSomething = True ; notSubfig = True    
                        if 'sub fig' in b[-1]: notSubfig = False
                        # look for some things that we are ignoring
                        # NOTE: some of this is redundant!!
                        if b[-1] == 'NotSure' or b[-1] == 'no label': # not sure then don't tag it!
                            notSubfig = False
                        else:
                            if labslabs[LABELS.index(b[-1])] == -1: notSubfig = False
                        if ignore_mathformula and 'math' in b[-1]: notSubfig = False
                        #if '1938ApJ____87__559S_p5' in ff: import sys; sys.exit()
                        if notSubfig or ('sub fig' in b[-1] and (scount <= sfcount) and (ccount == 0) and b[-1] != 'no label'): # this is overly convoluted
                            fo.write("\t<object>\n")
                            diagLab = ''
                            if (scount <= sfcount) and (ccount == 0): # call captions sub-figs
                                if b[-1] == 'sub fig caption':
                                    fo.write("\t\t<name>"+'figure caption'+"</name>\n") 
                                    diagLab = 'figure caption'
                                else:
                                    fo.write("\t\t<name>"+b[-1]+"</name>\n") 
                                    diagLab = b[-1]
                            else:
                                fo.write("\t\t<name>"+b[-1]+"</name>\n")
                                diagLab = b[-1]
                            fo.write("\t\t<bndbox>\n")
                            # update bounding boxes
                            #l = b[-1]
                            #x1min = b[0]*fracx; y1min = b[1]*fracy; x1max = b[2]*fracx; y1max = b[3]*fracy # inside?
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
                                            isOverlapping = (x1min <= x2max and x2min <= x1max and y1min <= y2max and y2min <= y1max)
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


                            b = trueBoxOut[0]
                            xmin = max([b[0]*1.0/d['w'].values[0]*IMAGE_W-xborder,0]) # have to rescale to output image size
                            xmax = min([(b[0]+b[2])*1.0/d['w'].values[0]*IMAGE_W+xborder,IMAGE_W])
                            ymin = max([b[1]*1.0/d['h'].values[0]*IMAGE_H-yborder,0])
                            ymax = min([(b[1]+b[3])*1.0/d['h'].values[0]*IMAGE_H+yborder,IMAGE_H])
                            fo.write("\t\t\t<xmin>" + str(int(round(xmin))) + "</xmin>\n")
                            fo.write("\t\t\t<ymin>" + str(int(round(ymin))) + "</ymin>\n")
                            fo.write("\t\t\t<xmax>" + str(int(round(xmax))) + "</xmax>\n")
                            fo.write("\t\t\t<ymax>" + str(int(round(ymax))) + "</ymax>\n")
                            fo.write("\t\t</bndbox>\n")    
                            fo.write("\t</object>\n") 
                            if plot_diagnostics:
                                # orig
                                xmin1 = xmin; ymin1 = ymin; xmax1 = xmax; ymax1 = ymax
                                b = borig
                                xmin = max([b[0]*1.0/d['w'].values[0]*IMAGE_W-xborder,0]) # have to rescale to output image size
                                xmax = min([(b[0]+b[2])*1.0/d['w'].values[0]*IMAGE_W+xborder,IMAGE_W])
                                ymin = max([b[1]*1.0/d['h'].values[0]*IMAGE_H-yborder,0])
                                ymax = min([(b[1]+b[3])*1.0/d['h'].values[0]*IMAGE_H+yborder,IMAGE_H])
                                #cv.rectangle(imgDiagResize, (round(xmin), round(ymin)), (round(xmax),round(ymax)), (0, 255, 255), 1)   
                                #ax[0].text(xmin, ymax, diagLab,bbox=dict(facecolor='blue'))
                                for b in captionBox:
                                    xmin = max([b[0]*1.0/d['w'].values[0]*IMAGE_W-xborder,0]) # have to rescale to output image size
                                    xmax = min([(b[0]+b[2])*1.0/d['w'].values[0]*IMAGE_W+xborder,IMAGE_W])
                                    ymin = max([b[1]*1.0/d['h'].values[0]*IMAGE_H-yborder,0])
                                    ymax = min([(b[1]+b[3])*1.0/d['h'].values[0]*IMAGE_H+yborder,IMAGE_H])
                                    cv.rectangle(imgDiagResize, (round(xmin), round(ymin)), (round(xmax),round(ymax)), (125, 255, 0), 1)   
                                cv.rectangle(imgDiagResize, (round(xmin1), round(ymin1)), (round(xmax1),round(ymax1)), (255, 0, 0), 1)  
                                #ax1.text(x.numpy(), y.numpy(), np.array(LABELS)[myclasses][i], bbox=dict(facecolor=colorLabel))
                                cv.putText(imgDiagResize,diagLab,(round(xmax1),round(ymax1)), cv.FONT_HERSHEY_SIMPLEX, 0.25, (255,0,0))



            fo.write("</annotation>\n")
            fo.close() 
            if plot_diagnostics:
                Image.fromarray(imgDiagResize).save(diagnostics_file + 'orig_ann/' + ff + '.png')
                imgDiag.close()
                del imgDiagResize
                #import sys; sys.exit()
            #if '4a809d73-8fe3-451a-9ff9-5e0a6a48a3d5' in f: import sys; sys.exit()
            #if '1fef1f64-8f8f-4da9-ab97-d26183d5111e' in f: 
            #    import sys; sys.exit()
            #if icc > 1: import sys; sys.exit()
