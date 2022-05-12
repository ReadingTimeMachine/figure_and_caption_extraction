####### Locations of general storage #######

# 0. Where are article PDFs stored?
full_article_pdfs_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/ADSDownloads/pdfs/'
# NOTE: the default is for PDFs but it will also look for individual pages in .bmp, .jpg, and .jpeg file formats

# 1. where to store JPEGs of individual pages from PDF articles?
#images_jpeg_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Pages/RandomSingleFromPDFIndexed/'
images_jpeg_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Pages/RandomSingleFromPDFIndexed/'

# 2. where to store OCR & image processing results?
#ocr_results_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/OCR_processing/'
ocr_results_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/OCR_processing/'

# 3. where should we store generated features and annotations?
save_binary_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/StoredFeatures/MegaYolo/'

# 4. where are MakeSense.ai annotations stored (as .csv files)
#make_sense_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Annotations/MakeSenseAnnotations/'
make_sense_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Annotations/MakeSenseAnnotations/'

# 5. What jar file for pdffigures2?
#pdffigures_jar_path = '/Users/jillnaiman/Downloads/ScanBank/bin/pdffigures2-assembly-0.1.0.jar' # use scanbank's
pdffigures_jar_path = '/Users/jnaiman/figure_and_caption_extraction/bin/pdffigures2-assembly-0.1.0.jar' # use ours

# 6. Where to store weights
#save_weights_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/'
save_weights_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/'

# X. What is a good temporary storage directory?
tmp_storage_dir = '/Users/jnaiman/Downloads/tmp/'

# 7. where to store results/metric results?
#metric_results_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/MetricsResults/'
metric_results_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/MetricsResults/'

# where to save numbers? these will be used by the paper
#save_table_dats_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/paper1/tables/tolatex/'
save_table_dats_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/paper1/tables/tolatex/'
# and figures?
#save_figures_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/paper1/figures/'
save_figures_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/paper1/figures/'


# X. Diagnostic dir
##mega_yolo_diag_dir = '/Users/jillnaiman/scienceDigitization/figure_localization/mega_yolo_model/diagnostics/'


## Some google directories
##main_google_dir = "/content/gdrive/My Drive/Colab Notebooks/scienceDigitization/"


####### Parameters for pipeline #########

# --------- Global Parameters ---------

# number of processers if you wanna do something in parallel?
nProcs = 6

# ------ OCR & Image Processing Parameters -----

# How many random pages to grab in one go? See paper for timing information
nRandom_ocr_image = 1500
# pulled from full_article_pdfs_dir

# do you want to use a list of files to process?
ocr_list_file = None #'/Users/jillnaiman/Downloads/tmp/presavelist.txt' # (see copy in misc folder)
 # set to none if no and you want to pull randomly
# will overwrite nRandom_ocr_image
# 2 columns: filename, pageNum (filename is the full path to the PDF file)

# default name for ocr-processing & image results -- take # will increase each time this is run
pickle_file_head = 'full_ocr_newPDFs_TIFF_take'
# NOTE: delete all these files if you want to start over (in ocr_results_dir)

# ------- Annotation Processing & Feature Generation ---------

# default filename for annotations directory
ann_name = 'yolo_'
# sometimes annotations have gone wrong, before or after processing -- where is the list of this? (in make_sens_dir)
bad_skews_file = 'more_bad_ann.csv' # see copy in misc folder
# 'filename' and then a list w/o any file extension or directory location

# what tags to ignore in creating annotations? for example math formulas or colorbars
#ignore_ann_list = ['math formula', 'table caption', 'colorbar', 'sub fig caption']
ignore_ann_list = ['table caption', 'colorbar', 'sub fig caption']

# do you want to plot out diagnostics (images) -- this will slow things down
plot_diagnostics = False # will plot to tmp directory + '/tmpAnnDiags/'

# features list -- see paper for more details
feature_list = ['grayscale','fontsize','carea boxes','paragraph boxes',
                'fraction of numbers in a word','fraction of letters in a word',
                'punctuation','x_decenders','x_ascenders','text angles', 
                'word confidences','Spacy POS','Spacy TAGs','Spacy DEPs']

# do we want to invert grayscale + other features or not?
feature_invert = True

# check for NaN's?  This slows things down, but is probably good for you :D
check_nans = True
# check that we can parse everything OK?
check_parse = True

# what type of file do you want to store TFRecords is HIGHLY recommended
astype = 'tfrecord'

# what about splits?
train_per = 0.75
valid_per = 0.15
test_per = 0.10

# ------  Yolo Parameters ---------

IMAGE_W = 512
IMAGE_H = 512

version = 'l' # start small, 'x' won't fit otn google collab
width_vec = [0.50, 0.75, 1.0, 1.25]
depth_vec = [0.33, 0.67, 1.0, 1.33]
versions = ['s', 'm', 'l', 'x']
threshold = 0.3
max_boxes = 150


# -------- Post Processing ----------

n_folds_cv = 5
weightsFileDir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/mega_yolo/saved_weights/20211111_model1/'
weightsFile = 'training_1model1_model_l0.17215717.h5' # figure/table, fig/table captions
benchmark = False # full run or benchmark run?

scoreminVec = [0.1, 0.5, 0.95] #   = box_conf * box_class_prob => pick with CV!!!
iouminVec = [0.1, 0.25, 0.6, 0.7, 0.75, 0.80, 0.9, 0.95] # 0.6 and 0.8 are from the ICDAR usual comparisons 

store_diagnostics = False

# from optimization: kh,kv
kpar = (7,3)
kparrot = (3,7)
# for figuring out if we are rotated -- what is our cut for word confidences?
ccut_rot = 90
# for finding figs
ccut_ocr_figcap = 20
# len text
len_text = 5
len_text1 = 5
len_text2 = 7
# blur
blurKernel = (3,3) # this or 9,9?

# overlap or only centers overlapping for found boxes cleaning?
found_overlap = 'overlap'
true_overlap = 'center' # what about for true boxes?

# for fuzzy-search of captions
keyWords = ['(FIG){e<=1}','(FIG.){e<=1}', '(FIGURE){e<=2}', '(PLATE){e<=2}']
lenMin = [   -1,               -1,           -1,                 5]
# how far in to look?
lookLength = 3
