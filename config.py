####### Locations of general storage #######

# 0. Where are article PDFs stored?
full_article_pdfs_dir = '/Users/jillnaiman/tmpADSDownloads/pdfs/'
# NOTE: the default is for PDFs but it will also look for individual pages in .bmp, .jpg, and .jpeg file formats

# 1. where to store JPEGs of individual pages from PDF articles?
images_jpeg_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Pages/RandomSingleFromPDFIndexed/'

# 2. where to store OCR & image processing results?
ocr_results_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/OCR_processing/'

# 3. where should we store generated features and annotations?
save_binary_dir = '/Users/jillnaiman/MegaYolo/binaries/'

# 4. where are MakeSense.ai annotations stored (as .csv files)
make_sense_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Annotations/MakeSenseAnnotations/'

# X. What is a good temporary storage directory?
tmp_storage_dir = '/Users/jillnaiman/Downloads/tmp/'



####### Parameters for pipeline #########

# --------- Global Parameters ---------

# number of processers if you wanna do something in parallel?
nProcs = 6

# ------ OCR & Image Processing Parameters -----

# How many random pages to grab in one go? See paper for timing information
nRandom_ocr_image = 1500
# pulled from full_article_pdfs_dir

# do you want to use a list of files to process?
ocr_list_file = None #'/Users/jillnaiman/Downloads/tmp/presavelist.txt'
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
bad_skews_file = 'more_bad_ann.csv'
# 'filename' and then a list w/o any file extension or directory location


# ------  Yolo Parameters ---------

IMAGE_W = 512
IMAGE_H = 512

