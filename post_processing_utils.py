from glob import glob
import xml.etree.ElementTree as ET
from scipy import stats
import numpy as np

def parse_annotations_to_labels(classDir_main_to, testListFile, benchmark=False):
    # from test file instead (from Google Collab splits)
    if not benchmark:
        with open(testListFile, 'r') as f:
            fl = f.readlines()
    else: # grab full list from directory
        fl = glob.glob(classDir_main_to+'*xml')
    
    annotations = []
    for f in fl:
        if len(glob(classDir_main_to + f.split('/')[-1].strip())) > 0: # check for file exists --> bug checking
            annotations.append(classDir_main_to + f.split('/')[-1].strip())
    annotations2 = glob(classDir_main_to + '*')
    # sort
    annotations = np.unique(annotations).tolist()
    annotations.sort()

    # NEXT: do a quick test run-through of the data generator for train/test splits
    X_full = np.array(annotations)
    Y_full_str = np.array([]) # have to loop and give best guesses for the pages that have multiple images/classes in them
    slabels = []
    for X in X_full:
        tree = ET.parse(X)
        tags = []
        for elem in tree.iter(): 
            if 'object' in elem.tag or 'part' in elem.tag:                  
                for attr in list(elem):
                    if 'name' in attr.tag:
                        if attr.text is not None:
                            tags.append(attr.text)
                            slabels.append(attr.text)
        if len(tags)>0: 
            modeClass = stats.mode(tags).mode[0] # most frequent class that pops up on this page
            Y_full_str = np.append(Y_full_str, modeClass) # class in string

    # NOTE: you need the full range of annotions to get ALL the labels:
    Y_full_str2 = np.array([]) # have to loop and give best guesses for the pages that have multiple images/classes in them
    slabels2 = []
    for X in annotations2:
        tree = ET.parse(X)
        tags = []
        for elem in tree.iter(): 
            if 'object' in elem.tag or 'part' in elem.tag:                  
                for attr in list(elem):
                    if 'name' in attr.tag:
                        if attr.text is not None:
                            tags.append(attr.text)
                            slabels2.append(attr.text)
        if len(tags) > 0:
            modeClass = stats.mode(tags).mode[0] # most frequent class that pops up on this page
            Y_full_str2 = np.append(Y_full_str2, modeClass) # class in string

    LABELS = np.unique(slabels2).tolist()
    CLASS = len(LABELS)
    #if use_only_one_class: CLASS=1

    # strings to integers
    Y_full = []
    labels = np.arange(len(LABELS))

    for i in range(len(Y_full_str)):
        Y_full.append( labels[np.array(LABELS) == Y_full_str[i]][0] +1 ) # 0 means unlabeled data
        if len(labels[np.array(LABELS) == Y_full_str[i]]) > 1:
            print('We have an issue!!')
            import sys
            sys.exit()

    Y_full = np.array(Y_full)
    
    return LABELS, labels, slabels, CLASS, annotations, Y_full