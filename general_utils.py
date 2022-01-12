import numpy as np
import xml.etree.ElementTree as ET
import os
from glob import glob
import shutil

def isRectangleOverlap(R1, R2):
    if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
        return False
    else:
        return True

def iou_orig(x1, y1, w1, h1, x2, y2, w2, h2, return_individual = False): 
    '''
    Calculate IOU between box1 and box2

    Parameters
    ----------
    - x, y : box ***center*** coords
    - w : box width
    - h : box height
    - return_individual: return intersection, union and IOU? default is False
    
    Returns
    -------
    - IOU
    '''   
    xmin1 = x1 - 0.5*w1
    xmax1 = x1 + 0.5*w1
    ymin1 = y1 - 0.5*h1
    ymax1 = y1 + 0.5*h1
    xmin2 = x2 - 0.5*w2
    xmax2 = x2 + 0.5*w2
    ymin2 = y2 - 0.5*h2
    ymax2 = y2 + 0.5*h2
    interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    inter = interx * intery
    union = w1*h1 + w2*h2 - inter
    iou = inter / (union + 1e-6)
    if not return_individual:
        return iou
    else:
        return inter, union, iou

# for parsing annotations
def parse_annotation(split_file_list, labels, feature_dir = '',
                     annotation_dir = '',
                     parse_pdf = False, use_only_one_class=False, 
                    IMAGE_W=512, IMAGE_H=512, debug=False, check_for_file=True):
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
                if check_for_file:
                    if not os.path.isfile(iname):
                        iname = iname[:iname.rfind('.')]
                        if debug: print(iname)
                        iname = glob(iname+'*')[0]
                imgs_name.append(iname)

            if 'width' in elem.tag:
                w = int(elem.text)
            if 'height' in elem.tag:
                h = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:                  
                box = np.zeros((5))
                #yesTag = True
                for attr in list(elem):
                    if 'name' in attr.tag:
                        if attr.text is not None:
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

    
    
    
# a = vector with (xc,yc)
# b = vector with (xc,yc)
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))


# save tmp binaries for this
def create_destroy_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    # delete and remake
    shutil.rmtree(dirs)
    os.makedirs(dirs)