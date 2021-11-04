####### Locations of general storage #######

# 0. Where are article PDFs stored?
full_article_pdfs_dir = '/Users/jillnaiman/tmpADSDownloads/pdfs/'
# NOTE: the default is for PDFs but it will also look for individual pages in .bmp, .jpg, and .jpeg file formats

# 1. where to store JPEGs of individual pages from PDF articles?
images_jpeg_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/Pages/RandomSingleFromPDFIndexed/'

# 2. where to store OCR & image processing results?
ocr_results_dir = '/Users/jillnaiman/Dropbox/wwt_image_extraction/FigureLocalization/OCR_processing/'

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
ocr_list_file = '/Users/jillnaiman/Downloads/tmp/presavelist.txt'
 # set to none if no and you want to pull randomly
# will overwrite nRandom_ocr_image
# 2 columns: filename, pageNum (filename is the full path to the PDF file)

# default name for ocr-processing & image results -- take # will increase each time this is run
pickle_file_head = 'full_ocr_newPDFs_TIFF_take'
# NOTE: delete all these files if you want to start over (in ocr_results_dir)

