import numpy as np
import pandas as pd
from general_utils import isRectangleOverlap, iou_orig

from collections import Counter
import sys

# from other metrics: https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/src/evaluators/pascal_voc_evaluator.py
def calculate_ap_every_point(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def calculate_ap_11_point_interp(rec, prec, recall_vals=11):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, recall_vals)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / len(recallValues)
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]


# calculations from: https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/src/evaluators/pascal_voc_evaluator.py
def new_calcs(gt_boxes, det_boxes, det_labels, det_scores, 
              iou_thresholds, fname, gt_classes_only,
              method = 'EVERY_POINT_INTERPOLATION', generate_table = True, 
             save_fp = None, image_shape=(512.,512.)):#, return_tf_pairs=False):
    have_an_fp = False
    ret = {}; retOut = {}
    # Get classes of all bounding boxes separating them by classes
    ##gt_classes_only = []
    classes_bbs = {}
    for bb in gt_boxes:
        c = int(bb[-1]-1) #bb.get_class_id()
        ##gt_classes_only.append(c)
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['gt'].append(bb)

    for bb,ll in zip(det_boxes,det_labels):
        c = ll #bb.get_class_id()
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['det'].append(bb)

    img = None
    # Precision x Recall is obtained individually by each class
    for c, v in classes_bbs.items():
        iou_threshold = iou_thresholds[c]
        # Report results only in the classes that are in the GT
        if c not in gt_classes_only:
            continue
        npos = len(v['gt'])
        # sort detections by decreasing confidence
        dects = [a for a in sorted(v['det'], key=lambda bb: det_scores, reverse=True)]
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        #FN = np.zeros(len(dects))
        # create dictionary with amount of expected detections for each image
        detected_gt_per_image = Counter([fname for bb in gt_boxes])
        for key, val in detected_gt_per_image.items():
            detected_gt_per_image[key] = np.zeros(val)
        # print(f'Evaluating class: {c}')
        dict_table = {
            'image': [],
            'confidence': [],
            'TP': [],
            'FP': [],
            'FN': [],
            'acc TP': [],
            'acc FP': [],
            'precision': [],
            'recall': []
        }
        # Loop through detections
        for (idx_det, det),dscore in zip(enumerate(dects),det_scores):
            img_det = fname #det.get_image_name()
            w2,h2 = det[2]-det[0],det[3]-det[1]
            x2,y2 = det[0]+0.5*w2, det[1]+0.5*h2
            if generate_table:
                dict_table['image'].append(img_det)
                dict_table['confidence'].append(f'{100*dscore:.2f}%')
                #dict_table['confidence'].append(f'{100*det.get_confidence():.2f}%')

            # Find ground truth image
            #gt = [gt for gt in classes_bbs[c]['gt'] if gt.get_image_name() == img_det]
            gt = [gt for gt in classes_bbs[c]['gt'] if fname == img_det]
            # Get the maximum iou among all detectins in the image
            iouMax = sys.float_info.min
            # Given the detection det, find ground-truth with the highest iou
            g = None
            for j, g in enumerate(gt):
                # print('Ground truth gt => %s' %
                #       str(g.get_absolute_bounding_box(format=BBFormat.XYX2Y2)))
                #iou = BoundingBox.iou(det, g)
                w1,h1 = g[2]-g[0],g[3]-g[1]
                x1,y1 = g[0]+0.5*w1, g[1]+0.5*h1
                iou = iou_orig(x1,y1,w1,h1, x2,y2,w2,h2)

                if iou > iouMax:
                    iouMax = iou
                    id_match_gt = j
                #print('iou=',iou)
            # Assign detection as TP or FP
            #print('iouMax = ', iouMax)
            if iouMax < 0:
                print('iouMax in new calc is < 0: ',iouMax)
            elif iouMax == 0:
                print('iouMax IS 0')
            if iouMax >= iou_threshold: 
                # gt was not matched with any detection
                if detected_gt_per_image[img_det][id_match_gt] == 0:
                    TP[idx_det] = 1  # detection is set as true positive
                    detected_gt_per_image[img_det][
                        id_match_gt] = 1  # set flag to identify gt as already 'matched'
                    # print("TP")
                    if generate_table:
                        dict_table['TP'].append(1)
                        dict_table['FP'].append(0)
                else:
                    FP[idx_det] = 1  # detection is set as false positive
                    #print('or here fp')
                    if generate_table:
                        dict_table['FP'].append(1)
                        dict_table['TP'].append(0)
                    if save_fp is not None: 
                        if img is None:
                            img = np.array(Image.open(images_pulled_dir+fname+'.jpeg').convert('RGB'))
                            fracx = img.shape[1]/image_shape[0]; fracy = img.shape[0]/image_shape[1] 
                        # plot found
                        c1 = (round(det[0]*fracx),round(det[1]*fracy)); 
                        c2 = (round(det[2]*fracx), round(det[3]*fracy))
                        cv.rectangle(img,c1,c2, (255,0,0), 5)
                        # plot true
                        if g is not None:
                            c1 = (round(g[0]*fracx),round(g[1]*fracy)); 
                            c2 = (round(g[2]*fracx), round(g[3]*fracy))
                            cv.rectangle(img,c1,c2, (0,0,255), 3)
                        have_an_fp = True
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= iou_threshold.
            else:
                FP[idx_det] = 1  # detection is set as false positive
                #print('here fp')
                if generate_table:
                    dict_table['FP'].append(1)
                    dict_table['TP'].append(0)
                if save_fp is not None: 
                    if img is None:
                        img = np.array(Image.open(images_pulled_dir+fname+'.jpeg').convert('RGB'))
                        fracx = img.shape[1]/image_shape[0]; fracy = img.shape[0]/image_shape[1] 
                    # plot found
                    c1 = (round(det[0]*fracx),round(det[1]*fracy)); 
                    c2 = (round(det[2]*fracx), round(det[3]*fracy))
                    cv.rectangle(img,c1,c2, (255,0,0), 5)
                    # plot true
                    if g is not None:
                        c1 = (round(g[0]*fracx),round(g[1]*fracy)); 
                        c2 = (round(g[2]*fracx), round(g[3]*fracy))
                        cv.rectangle(img,c1,c2, (0,0,255), 3)
                    have_an_fp = True
                # print("FP")
        tphere = 0; fphere=0
        if len(dict_table['TP'])>0: 
            tphere = dict_table['TP'][-1]
            dict_table['TP'].append(0)
        if len(dict_table['FP'])>0: 
            fphere = dict_table['FP'][-1]
            dict_table['FP'].append(0)
        #dict_table['FN'].append(npos - tphere - fphere) # WRONG
        dict_table['FN'].append(-1) # placeholder
        #if len(dict_table['TP']) > 0: # we have some true positives
        #    # false negatives will be if there are "extra" boxes found
        #    dict_table['FN'].append(npos - dict_table['TP'][-1] - dict_table['FP'][-1])
        #else:
        #    dict_table['FN'].append(npos)
        #    dict_table['TP'].append(0)
        #    if len(dict_table['FP']) == 0: dict_table['FP'].append(0)
        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        with np.errstate(invalid='ignore'): # take care of zeros later
            rec = acc_TP / npos
        #if npos == 0: # no positive stuff
        #    rec = acc_TP/1.0
        # from this eq??
        ####acc_FN = npos-acc_TP
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        if generate_table:
            dict_table['acc TP'] = list(acc_TP)
            dict_table['acc FP'] = list(acc_FP)
            dict_table['precision'] = list(prec)
            dict_table['recall'] = list(rec)
            ###dict_table['acc FN'] = list(acc_FN)
            #table = pd.DataFrame(dict_table)
        else:
            table = None
        # Depending on the method, call the right implementation
        if method == 'EVERY_POINT_INTERPOLATION':
            [ap, mpre, mrec, ii] = calculate_ap_every_point(rec, prec)
        elif method == 'ELEVEN_POINT_INTERPOLATION':
            [ap, mpre, mrec, _] = calculate_ap_11_point_interp(rec, prec)
        else:
            Exception('method not defined')
        #print(ap)
        # add class result in the dictionary to be returned
        ret[c] = {
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP),
            'total FN': npos-np.sum(TP),
            'method': method,
            'iou': iou_threshold,
            #'table': table
        }
        retOut[c] = {'TP':np.sum(TP), 'FP':np.sum(FP), 
                     'FN':npos-np.sum(TP)-np.sum(FP), 'npos':npos, 
                     'year':int(fname[:4]), 'name':fname}#, 
                    #'AP':ap}
        # FN IS WrONG
        
    if save_fp is not None and have_an_fp: 
        Image.fromarray(img).save(save_fp+fname+'.jpeg')
        del img

    return retOut


# FP/FN/TP calcs:
def calc_metrics(truebox1, boxes_sq_in, labels_sq_in, scores_sq_in, LABELS,ioumin,
                years=[], iioumin=-1, iscoremin=-1, 
                TPyear = [], FPyear=[], totalTrueyear=[], FNyear=[], year=[],
                totalTruev=[], TPv=[], FPv=[],FNv=[], return_pairs = False, 
                use_review_calc = True, round_here=True):#, 
                #return_mAP=False, mAP_range=[0.5,0.95,0.5]):
    """
    truebox1: trueboxes in YOLO coords (typically 512x512) (xmin,ymin,xmax,ymax, LABEL_INDEX+1)
    boxes_sq: found boxes in YOLO coords (xmin,ymin,xmax,ymax)
    labels_sq: found box labels (LABEL_INDEX) -- NOTE no +1!
    scores_sq: found box score (0.0-1.0)
    LABELS: list of labels (strings), e.g. ['figure', 'figure caption', 'table']
    round_here: (True) -- since trueboxes are saved in increments of pixels, you 
                 generally want to force this on found boxes, otherwise, you can 
                have issues for very small boxes.
    XXXreturn_mAP: calculate and return map or not?
    XXXmAP_range: [start, stop, step size] -- default is for COCO/ICDAR challenges
    """
    
    if round_here:
        truebox1 = np.round(truebox1)
        boxes_sq_in = np.round(boxes_sq_in)
    
    # for heuristically-found labels, replace -2 tag with tag for figure caption
    labels_sq = labels_sq_in.copy()
    for il in range(len(labels_sq)):
        if labels_sq[il] == -2:
            labels_sq[il] = LABELS.index('figure caption')
    #print(' ----- ')
    #print(labels_sq_in,labels_sq)
    boxes_sq = boxes_sq_in.copy(); scores_sq = scores_sq_in.copy()
    
    # checks
    if iioumin == -1 and iscoremin != -1: 
        print('not supported for different params for iioumin & iscoremin')
        import sys; sys.exit()
    # other checks
    if iioumin == -1 and iscoremin == -1:
        totalTruev = np.zeros(len(LABELS))
        FNv = np.zeros(len(LABELS))
        FPv = np.zeros(len(LABELS))
        TPv = np.zeros(len(LABELS))
        if len(years)>0:
            totalTrueyear = np.zeros([len(years),len(LABELS)])
            FNyear = np.zeros([len(years),len(LABELS)])
            FPyear = np.zeros([len(years),len(LABELS)])
            TPyear = np.zeros([len(years),len(LABELS)])

    if use_review_calc:
        gt_classes_only = np.arange(len(LABELS))
        stats = []
        #for ibox in range(len(founds)):
            #found = founds[ibox]; true = trues[ibox]; fname = fnames[ibox]
            #gt_boxes = true; det_boxes = found[0]; det_labels = found[1]; det_scores = found[2]
            #stats.append(new_calcs(gt_boxes, det_boxes, det_labels, det_scores, 
            #                       [ioumin], fname, gt_classes_only))
        stats.append(new_calcs(truebox1, boxes_sq, labels_sq, scores_sq, 
                               np.repeat(ioumin,len(LABELS)), 
                               '1000_placeholder', gt_classes_only))

        # fill all arrays
        #print(stats)
        totalTrues11 = np.zeros(len(LABELS)).tolist()
        totalFounds = np.zeros(len(LABELS)).tolist()
        #print(totalTrues11)
        for t in truebox1:
            #print(int(t[-1]-1))
            totalTrues11[int(t[-1]-1)] += 1
        for f,l in zip(boxes_sq,labels_sq):
            totalFounds[int(l)] += 1
        #print(totalTrues11)
        #print(truebox1)
        #print('--')
        #TPall = np.zeros(len(LABELS)); FPall = np.zeros(len(LABELS)); 
        #FNall = np.zeros(len(LABELS)); allall = np.zeros(len(LABELS))
        #Npos = np.zeros(len(LABELS))
        #FpList = []
        
        # # return mAP?
        # if return_mAP:
        #     for ious in np.arange(mAP_range[0],mAP_range[1] + mAP_range[2],mAP_range[2]):
        #         statsMAP = new_calcs(truebox1, boxes_sq, labels_sq, scores_sq, 
        #                        np.repeat(ious,len(LABELS)), 
        #                        '1000_placeholder', gt_classes_only)
        #         #print(statsMAP)
            
        
        for s in stats:
            for l in range(len(LABELS)):
                if l in s:
                    if iioumin != -1:
                        TPv[l,iioumin,iscoremin] += s[l]['TP']
                        FPv[l,iioumin,iscoremin] += s[l]['FP']
                        #if s[l]['FP'] == 1: FpList.append(s[l]['name'])
                        #FNv[l,iioumin,iscoremin] += s[l]['FN'] # NOT RIGHT
                        FNv[l,iioumin,iscoremin] += max([totalTrues11[l]-totalFounds[l],0])
                        #Npos[l] += s[l]['npos']
                        #indstt = np.where(truebox1[-1]
                        totalTruev[l,iioumin,iscoremin] = totalTrues11[l]#
                                          #s[l]['TP'] + s[l]['FN']  
                    else:
                        TPv[l] += s[l]['TP']
                        FPv[l] += s[l]['FP']
                        FNv[l] += max([totalTrues11[l]-totalFounds[l],0])#s[l]['FN']
                        totalTruev[l] = totalTrues11[l]   
        # for years too if we have them
        # per year
        # years2 = []
        # for s in stats:
        #     for l in range(len(LABELS)):
        #         if l in s:
        #             years2.append(s[l]['year'])
        # years2 = np.unique(years2)
        # years2
        # TPyear2 = np.zeros([len(years2),len(LABELS)])
        # FPyear2 = np.zeros([len(years2),len(LABELS)])
        # FNyear2 = np.zeros([len(years2),len(LABELS)])
        # Nposyear2 = np.zeros([len(years2),len(LABELS)])
        if len(years)>0:
            for s in stats:
                for l in range(len(LABELS)):
                    if l in s:
                        if iioumin != -1:
                            TPyear[years==year,l,iioumin,iscoremin] += s[l]['TP']
                            FPyear[years==year,l,iioumin,iscoremin] += s[l]['FP']
                            FNyear[years==year,l,iioumin,iscoremin] += s[l]['FN']
                            #Nposyear2[years==year,l,iioumin,iscoremin] += s[l]['npos']
                            totalTrueyear[years==year,l,iioumin,iscoremin] += s[l]['TP'] + s[l]['FN']
                        else:
                            TPyear[years==year,l] += s[l]['TP']
                            FPyear[years==year,l] += s[l]['FP']
                            FNyear[years==year,l] += s[l]['FN']
                            totalTrueyear[years==year,l] += s[l]['TP'] + s[l]['FN']
                    
    if not use_review_calc:
        # make array of total number of boxes true/found, tag extra found with a FP tag
        # loop and find closest boxes
        true_found_index = []; true_found_labels = []; trueCaps = []; foundCaps = []; 
        for it,tbox in enumerate(truebox1): # if there really is a box
            #print(tbox)
            #print(' ')
            w2, h2 = tbox[2]-tbox[0], tbox[3]-tbox[1]
            x2, y2 = tbox[0]+0.5*w2, tbox[1]+0.5*h2
            # just for found captions
            if iioumin != -1:
                totalTruev[int(tbox[4]-1),iioumin,iscoremin] += 1
                if len(years)>0:
                    totalTrueyear[years==year,int(tbox[4]-1),iioumin,iscoremin] += 1                
            else:
                totalTruev[int(tbox[4]-1)] += 1
                if len(years) >0:
                    totalTrueyear[years==year,int(tbox[4]-1)] += 1 
            #print(totalTruev)

            trueCaps.append(tbox)
            # find greatest IOU -- literally there is a better algorithm here to do this
            iouMax = -10
            foundBox = False
            indFound = [it,-1]; labelsFound = [tbox[-1]-1, -1]; foundCapHere = []
            for ib,b in enumerate(boxes_sq):
                isOverlapping = isRectangleOverlap(tbox[:-1],b)
                w1, h1 = b[2]-b[0], b[3]-b[1]
                x1, y1 = b[0]+0.5*w1, b[1]+0.5*h1
                iou1 = iou_orig(x1,y1,w1,h1, x2,y2,w2,h2)
                # a win!
                if (iou1 > iouMax) and (iou1 >= ioumin) and isOverlapping: # check for overlap
                    iouMax = iou1
                    indFound[-1] = ib
                    labelsFound[-1] = labels_sq[ib]
                    #print('labelsFound=',labelsFound[-1])
                    # if tagged as a heurstically-only found caption, mark as caption
                    #if labels_sq[ib] == -2:
                    #    labelsFound[-1] = LABELS.index('figure caption')
                    foundCapHere = b
            true_found_index.append(indFound); true_found_labels.append(labelsFound); 
            if len(foundCapHere) > 0: foundCaps.append(foundCapHere)

        #print(true_found_index,true_found_labels,'here')
        # count
        # save pairs, unfound trues, miss-found founds
        true_found_pairs = []
        # ti = [true index, found index]
        for ti,tl in zip(true_found_index, true_found_labels): 
            ind = int(tl[0]) # index is true's label

            if ti[-1] == -1: # didn't find anything
                if iioumin != -1:
                    FNv[ind,iioumin,iscoremin] +=1
                    if len(years)>0:
                        FNyear[years==year,ind,iioumin,iscoremin] += 1
                else:
                    FNv[ind] += 1
                    if len(years)>0:
                        FNyear[years==year,ind] +=1
                # save as a true w/o a found
                true_found_pairs.append( (truebox1[ti[0]], -1) )
            # overlap of boxes, but wrong things -- count as FN for this true
            elif ti[-1] != -1 and tl[0] != tl[1]: 
                if iioumin != -1:
                    FNv[ind,iioumin,iscoremin] +=1
                    if len(years)>0:
                        FNyear[years==year,ind,iioumin,iscoremin] += 1
                else:
                    FNv[ind] += 1
                    if len(years)>0:
                        FNyear[years==year,ind] +=1
                # save as a true w/o a found
                true_found_pairs.append( (truebox1[ti[0]], -1) )
            elif ti[-1] != -1 and tl[0] == tl[1]: # found a box AND its the right one!
                if iioumin != -1:
                    TPv[ind,iioumin,iscoremin] +=1
                    if len(years)>0:
                        TPyear[years==year,ind,iioumin,iscoremin] += 1
                else:
                    TPv[ind] += 1
                    if len(years)>0:
                        TPyear[years==year,ind] +=1
                #try:
                true_found_pairs.append( (truebox1[ti[0]], (boxes_sq[ti[1]][0],
                                                            boxes_sq[ti[1]][1],
                                                            boxes_sq[ti[1]][2],
                                                            boxes_sq[ti[1]][3],
                                                            labels_sq[ti[1]])) )

        # do we have extra found boxes?
        #print('bbox, truecap', len(boxes_sq), len(trueCaps))
        #print('FPv here', FPv)
        #if len(boxes_sq) > len(trueCaps):
        #if True: # messy
        #print(true_found_index)
        #print('trufound',np.array(true_found_index)[:,1].tolist())
        for ib, b in enumerate(boxes_sq):
            if len(true_found_index) > 0: # we have some trues
                # but we don't have this particular found matched to a true
                if ib not in np.array(true_found_index)[:,1].tolist(): 
                    ind = int(labels_sq[ib]) # label will be found label -- mark as a FP for this label
                    #print('hi',labels_sq[ib])
                    # is this a heuristically found caption? if so -- tag it not with index -2, but cap
                    #if ind == -2:
                    #    ind = LABELS.index('figure caption')
                    #print('ind', ind)
                    if iioumin != -1:
                        FPv[ind,iioumin,iscoremin] +=1
                        if len(years)>0:
                            FPyear[years==year,ind,iioumin,iscoremin] += 1
                    else:
                        FPv[ind] += 1
                        if len(years)>0:
                            FPyear[years==year,ind] +=1
                    # mark as a found w/o a true
                    true_found_pairs.append( (-1, (boxes_sq[ib][0],
                                                   boxes_sq[ib][1],
                                                    boxes_sq[ib][2],
                                                   boxes_sq[ib][3],
                                                   labels_sq[ib])) )
            elif len(true_found_index) == 0: # there is nothing true, any founds are FP
                #print('no')
                ind = int(labels_sq[ib])
                #if ind == -2:
                #    ind = LABELS.index('figure caption')
                if iioumin != -1:
                    FPv[ind,iioumin,iscoremin] +=1
                    if len(years)>0:
                        FPyear[years==year,ind,iioumin,iscoremin] += 1
                else:
                    FPv[ind] += 1
                    if len(years)>0:
                        FPyear[years==year,ind] +=1
                # mark as a found w/o a true
                true_found_pairs.append( (-1, (boxes_sq[ib][0],
                                               boxes_sq[ib][1],
                                                boxes_sq[ib][2],
                                               boxes_sq[ib][3],
                                               labels_sq[ib])) )

               
    if len(years)>0:
        if not return_pairs:
            return totalTruev, TPv, FPv, FNv, totalTrueyear, TPyear, FPyear, FNyear
        else:
            return totalTruev, TPv, FPv, FNv, totalTrueyear, TPyear, FPyear, FNyear, true_found_pairs
    else:
        if not return_pairs:
            return totalTruev, TPv, FPv, FNv
        else:
            return totalTruev, TPv, FPv, FNv,true_found_pairs

        
def calc_base_metrics_allboxes_cv(LABELS,scoreminVec,iouminVec,
                                  truebox2,boxes_sq4,labels_sq4,scores_sq4,
                                  n_folds_cv=5, seed=None, return_FP_ind = False):#,
                                 #return_mAP=False, mAP_range=[0.5,0.95,0.5]):     
    TPs = np.zeros([len(LABELS), len(scoreminVec),len(iouminVec),n_folds_cv])
    #TPs = np.zeros([len(LABELS), len(iouminVec),len(scoreminVec),n_folds_cv])
    totalTrues = TPs.copy(); FPs = TPs.copy(); FNs = TPs.copy()

    # place randomly, unless reproducing
    np.random.seed(seed)
    rinds = np.random.randint(0,n_folds_cv, len(truebox2))
    
    fpind = []

    for iit in range(len(truebox2)): # an array of pages
        for iscore,scoremin in enumerate(scoreminVec):
            for iiou,ioumin in enumerate(iouminVec):
                # truebox2 is fig-caption pairs, unwrap for this
                tboxes = []
                for t in truebox2[iit]:
                    #for t in tt:
                    tboxes.append(t)
                #print(tboxes)
                #print(' ')
                # same same for found boxes
                bboxes = []; llabels = []; sscores = []
                for b,l,s in zip(boxes_sq4[iit],labels_sq4[iit],scores_sq4[iit]):
                    #for b,l,s in zip(bb,ll,ss):
                    if s >= scoremin:
                        bboxes.append(b); llabels.append(l); sscores.append(s)

                totalTruev1, TPv1, FPv1, FNv1 = calc_metrics(tboxes, bboxes, llabels, 
                                                             sscores, LABELS,ioumin)#,
                                                            #return_mAP=return_mAP,
                                                            #mAP_range=mAP_range)
                #print(FPv1)
                # only the "fake" index
                #totalTrues[:,iiou,iscore,rinds[iit]] += totalTruev1; 
                totalTrues[:,iscore,iiou,rinds[iit]] += totalTruev1; 
                TPs[:,iscore,iiou,rinds[iit]] += TPv1; 
                FPs[:,iscore,iiou,rinds[iit]] += FPv1; 
                FNs[:,iscore,iiou,rinds[iit]] += FNv1
                #TPs[:,iiou,iscore,rinds[iit]] += TPv1; 
                #FPs[:,iiou,iscore,rinds[iit]] += FPv1; 
                #FNs[:,iiou,iscore,rinds[iit]] += FNv1
                
    # total trues, are actually just for the whole thing -- could do this better!
    totalTrues = totalTrues[:,0,0,:]
                
    return TPs, FPs, FNs, totalTrues


# calc precision/recall and spreads
def calc_prec_rec_f1_cv(TPv,FPv,FNv,LABELS,scoreminVec,iouminVec):
    precision = np.zeros([len(LABELS), len(scoreminVec),len(iouminVec)])
    recall = np.zeros([len(LABELS), len(scoreminVec),len(iouminVec)])
    f1 = np.zeros([len(LABELS), len(scoreminVec),len(iouminVec)])

    # from CV
    precision_std = np.zeros([len(LABELS), len(scoreminVec),len(iouminVec)])
    recall_std = np.zeros([len(LABELS), len(scoreminVec),len(iouminVec)])
    f1_std = np.zeros([len(LABELS), len(scoreminVec),len(iouminVec)])

    for j in range(len(iouminVec)):
        for i in range(len(scoreminVec)):
            with np.errstate(invalid='ignore'): # take care of zeros later
                p = TPv[:,i,j]/(TPv[:,i,j]+FPv[:,i,j])*100
                p[TPv[:,i,j]+FPv[:,i,j]<=0] = 0
                r = TPv[:,i,j]/(TPv[:,i,j]+FNv[:,i,j])*100
                r[TPv[:,i,j]+FNv[:,i,j]<=0] = 0
                f = 2.0*(r*p)/(r+p)
            f[r+p <=0] = 0
            precision_std[:,i,j] = np.std(p,axis=1)
            recall_std[:,i,j] = np.std(r,axis=1)
            f1_std[:,i,j] = np.std(f,axis=1)
            p = np.mean(p, axis=1)
            r = np.mean(r, axis=1)
            f = np.mean(f, axis=1)

            precision[:,i,j] = p
            recall[:,i,j] = r
            f1[:,i,j] = f
    return precision, precision_std, recall, recall_std, f1, f1_std


def print_metrics_table(totalTrue,TP,FP,FN,
                        precision, precision_std, recall, recall_std,f1,f1_std,
                        LABELS, scoremin, n_folds_cv, ioumin_per_label):
    print('SCORE = ', scoremin, ' N_CV = ', n_folds_cv)

    labelsMetric = ['Metric']
    labelsMetric.extend(LABELS)
    spacing = '%-15s'
    strOut = ' '.join([spacing % (i,) for i in labelsMetric])
    print(strOut)

    out = ['iou cut']
    for i in range(len(LABELS)):
        out.append( str(ioumin_per_label[i]) )
    strOut = ' '.join([spacing % (i,) for i in out])
    print(strOut)

    out = ['# of objs']
    for i in range(len(LABELS)):
        out.append( str(totalTrue[i]) )
    strOut = ' '.join([spacing % (i,) for i in out])
    print(strOut)

    print('--------------------------------------------------------------------------------------------')

    out = ['TP']
    for i in range(len(LABELS)):
        out.append( str(round(TP[i]/totalTrue[i]*100,1))+'%' )
    strOut = ' '.join([spacing % (i,) for i in out])
    print(strOut)

    out = ['FP']
    for i in range(len(LABELS)):
        out.append( str(round(FP[i]/totalTrue[i]*100,1))+'%' )
    strOut = ' '.join([spacing % (i,) for i in out])
    print(strOut)

    out = ['FN']
    for i in range(len(LABELS)):
        out.append( str(round(FN[i]/totalTrue[i]*100,1))+'%' )
    strOut = ' '.join([spacing % (i,) for i in out])
    print(strOut)

    print('--------------------------------------------------------------------------------------------')
    out = ['Precision'] # accuracy of positive predictions => number of true positives out of all of the things we label as positive
    for i in range(len(LABELS)):
        #iind = np.where(iouminVec == ioumin_per_label[i])[0]
        #out.append( str(np.round(precision[i,sind,iind][0],1))+'+/-' +str(np.round(precision_std[i,sind,iind][0],1))+ '%' )
        out.append( str(np.round(precision[i],1))+'+/-' +str(np.round(precision_std[i],1))+ '%' )
    strOut = ' '.join([spacing % (i,) for i in out])
    print(strOut)

    out = ['Recall'] # true positive rate => number of true positives over all of the things that SHOULD be positive
    for i in range(len(LABELS)):
        #iind = np.where(iouminVec == ioumin_per_label[i])[0]
        out.append( str(np.round(recall[i],1))+'+/-' +str(np.round(recall_std[i],1))+ '%' )
    strOut = ' '.join([spacing % (i,) for i in out])
    print(strOut)

    out = ['F1']
    for i in range(len(LABELS)):
        #iind = np.where(iouminVec == ioumin_per_label[i])[0]
        out.append( str(np.round(f1[i],1))+'+/-' +str(np.round(f1_std[i],1))+ '%' )
    strOut = ' '.join([spacing % (i,) for i in out])
    print(strOut)
    
    
def get_years_dataframe(imgs_name,scoremin,ioumin,LABELS,
                       truebox3,boxes_sq5,labels_sq5,scores_sq5,
                       fake_years=False):
    
    if not fake_years:
        years = []
        for n in imgs_name:
            years.append(n.split('/')[-1][:4])
    else:
        years = np.random.randint(1900,2000,len(imgs_name))
    years_u = np.unique(years).astype('int')
        
    TPyear = np.zeros([len(years_u),len(LABELS)])
    FPyear = np.zeros([len(years_u),len(LABELS)])
    FNyear = np.zeros([len(years_u),len(LABELS)])
    TTyear = np.zeros([len(years_u),len(LABELS)])

    for t,b,l,s,y in zip(truebox3,boxes_sq5,labels_sq5,scores_sq5,years):
        TPv2, FPv2, FNv2, totalTruev2 = calc_base_metrics_allboxes_cv(LABELS,
                                                                      [scoremin],
                                                                      [ioumin],
                                                                      [t],[b],[l],[s],
                                                                      n_folds_cv=1)
        # then fill
        ind = np.where(years_u == int(y))[0]
        TPyear[ind,:] += TPv2.flatten()
        FPyear[ind,:] += FPv2.flatten()
        FNyear[ind,:] += FNv2.flatten()
        TTyear[ind,:] += totalTruev2.flatten()
        
    df = pd.DataFrame({'years':years_u})
    df['years'] = pd.to_datetime(df['years'],format="%Y")

    # total each year
    for il,l in enumerate(LABELS):
        df['total:'+l] = TTyear[:,il]

    # true positive per year
    for il,l in enumerate(LABELS):
        colname = 'TP:' + l
        df[colname] = TPyear[:,il]

    # false positive each year
    for il,l in enumerate(LABELS):
        colname = 'FP:' + l
        df[colname] = FPyear[:,il]

    # false negative each year
    for il,l in enumerate(LABELS):
        colname = 'FN:' + l
        df[colname] = FNyear[:,il]
        
    return df


# map calculation
def calc_AP(truebox3,boxes_sq5,labels_sq5, scores_sq5,LABELS, 
            scoreMin = [0.1], iou_mAP_coco_range =[0.5,0.95,0.05]):
    # iou ranges -- default is COCO
    ious_mAP = np.arange(iou_mAP_coco_range[0],
                     iou_mAP_coco_range[1] + iou_mAP_coco_range[2],
                     iou_mAP_coco_range[2])
    
    TPv, FPv, FNv, totalTruev = calc_base_metrics_allboxes_cv(LABELS,scoreMin,ious_mAP,
                                                  truebox3,boxes_sq5,labels_sq5, 
                                                  scores_sq5,n_folds_cv=1)
    precision, precision_std, recall, \
      recall_std, f1, f1_std = calc_prec_rec_f1_cv(TPv,FPv,FNv,
                                                   LABELS,scoreMin,
                                                   ious_mAP)

    precision = precision.reshape((len(LABELS),len(ious_mAP)))
    recall = recall.reshape((len(LABELS),len(ious_mAP)))
    # ap per class:
    apOut = []
    for l in range(len(LABELS)):
        [ap, mpre, mrec, ii] = calculate_ap_every_point(recall[l,:]/100., 
                                                        precision[l,:]/100.)
        apOut.append(ap)
        
    return apOut
