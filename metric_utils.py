import numpy as np
from general_utils import isRectangleOverlap, iou_orig

# FP/FN/TP calcs:
def calc_metrics(truebox1, boxes_sq, labels_sq, scores_sq, LABELS,ioumin,
                years=[], iioumin=-1, iscoremin=-1, 
                TPyear = [], FPyear=[], totalTrueyear=[], FNyear=[], year=[],
                totalTruev=[], TPv=[], FPv=[],FNv=[], return_pairs = False):
    """
    truebox1: trueboxes in YOLO coords (typically 512x512) (xmin,ymin,xmax,ymax, LABEL_INDEX+1)
    boxes_sq: found boxes in YOLO coords (xmin,ymin,xmax,ymax)
    labels_sq: found box labels (LABEL_INDEX) -- NOTE no +1!
    scores_sq: found box score (0.0-1.0)
    LABELS: list of labels (strings), e.g. ['figure', 'figure caption', 'table']
    """
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

    # make array of total number of boxes true/found, tag extra found with a FP tag
    # loop and find closest boxes
    true_found_index = []; true_found_labels = []; trueCaps = []; foundCaps = []; #iouSave = []
    for it,tbox in enumerate(truebox1): # if there really is a box
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
            if (iou1 > iouMax) and (iou1 > ioumin) and isOverlapping: # check for overlap
                iouMax = iou1
                indFound[-1] = ib
                labelsFound[-1] = labels_sq[ib]
                foundCapHere = b
                #print('yes')
        true_found_index.append(indFound); true_found_labels.append(labelsFound); 
        if len(foundCapHere) > 0: foundCaps.append(foundCapHere)
        
    # count
    # save pairs, unfound trues, miss-found founds
    true_found_pairs = []
    for ti,tl in zip(true_found_index, true_found_labels): # ti = [true index, found index]
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
        elif ti[-1] != -1 and tl[0] != tl[1]: # overlap of boxes, but wrong things -- count as FN for this true
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
            true_found_pairs.append( (truebox1[ti[0]], (boxes_sq[ti[1]][0],boxes_sq[ti[1]][1],
                                                    boxes_sq[ti[1]][2],boxes_sq[ti[1]][3],labels_sq[ti[1]])) )
#             except:
#                 print('ti',ti)
#                 print(' ')
#                 print('truebox',truebox1)
#                 print(' ')
#                 print('boxes_sq', boxes_sq)
#                 print(' ')
#                 print('labels_sq', labels_sq)
#                 import sys; sys.exit()
            
    # do we have extra found boxes?
    if len(boxes_sq) > len(trueCaps):
        for ib, b in enumerate(boxes_sq):
            if len(true_found_index) > 0: # we have some trues
                if ib not in np.array(true_found_index)[:,1].tolist(): # but we don't have this particular found matched to a true
                    ind = labels_sq[ib] # label will be found label -- mark as a FP for this label
                    if iioumin != -1:
                        FPv[ind,iioumin,iscoremin] +=1
                        if len(years)>0:
                            FPyear[years==year,ind,iioumin,iscoremin] += 1
                    else:
                        FPv[ind] += 1
                        if len(years)>0:
                            FPyear[years==year,ind] +=1
                    # mark as a found w/o a true
                    true_found_pairs.append( (-1, (boxes_sq[ib][0],boxes_sq[ib][1],
                                                    boxes_sq[ib][2],boxes_sq[ib][3],labels_sq[ib])) )
            elif len(true_found_index) == 0: # there is nothing true, any founds are FP
                ind = labels_sq[ib]
                if iioumin != -1:
                    FPv[ind,iioumin,iscoremin] +=1
                    if len(years)>0:
                        FPyear[years==year,ind,iioumin,iscoremin] += 1
                else:
                    FPv[ind] += 1
                    if len(years)>0:
                        FPyear[years==year,ind] +=1
                # mark as a found w/o a true
                true_found_pairs.append( (-1, (boxes_sq[ib][0],boxes_sq[ib][1],
                                                boxes_sq[ib][2],boxes_sq[ib][3],labels_sq[ib])) )

               
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
                                  n_folds_cv=5):     
    TPs = np.zeros([len(LABELS), len(scoreminVec),len(iouminVec),n_folds_cv])
    #TPs = np.zeros([len(LABELS), len(iouminVec),len(scoreminVec),n_folds_cv])
    totalTrues = TPs.copy(); FPs = TPs.copy(); FNs = TPs.copy()

    # place randomly
    rinds = np.random.randint(0,n_folds_cv, len(truebox2))

    for iit in range(len(truebox2)):
        for iscore,scoremin in enumerate(scoreminVec):
            for iiou,ioumin in enumerate(iouminVec):
                # truebox2 is fig-caption pairs, unwrap for this
                tboxes = []
                for t in truebox2[iit]:
                    #for t in tt:
                    tboxes.append(t)
                # same same for found boxes
                bboxes = []; llabels = []; sscores = []
                for b,l,s in zip(boxes_sq4[iit],labels_sq4[iit],scores_sq4[iit]):
                    #for b,l,s in zip(bb,ll,ss):
                    if s >= scoremin:
                        bboxes.append(b); llabels.append(l); sscores.append(s)

                totalTruev1, TPv1, FPv1, FNv1 = calc_metrics(tboxes, bboxes, llabels, 
                                                             sscores, LABELS,ioumin)
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

    #for j in range(len(iouminVec)):
    #    for i in range(len(scoreminVec)):
    for i in range(len(iouminVec)):
        for j in range(len(scoreminVec)):
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