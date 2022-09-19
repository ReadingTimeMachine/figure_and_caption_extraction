# Please excuse our dust!

This code is very much in Beta and needs a through deep clean. 

This work has been excepted to [TPDL 2022](https://link.springer.com/book/10.1007/978-3-031-16802-4) as [Figure and Figure Caption Extraction for Mixed Raster and Vector PDFs: Digitization of Astronomical Literature with OCR Features](https://arxiv.org/abs/2209.04460).

Writing of talk (and, let's be honest, eating of Italian food) took precedence, but expect cleaner code mid-October.

## Order

 1. change paths in config.py for system
 2. `ocr_and_image_processing_batch.py` -- runs the OCR and pdfmining to get raw data from pages
 3. `pull_check_makesense.ipynb` -- uses a current model to "guess" boxes, and prepares them to check with MakeSense.ai
 4. `process_annotations_and_generate_features_batch.py` -- process annotations after downloading (as a csv file) from MakeSense.ai, ~~generate features as well if you want~~
 5. `generate_features_only.py` -- generates features in batchs for certain feature sets, saves them in tfrecords format
 6. `mega_yolo_train_tfrecords.ipynb` -- to be run on the cloud (set up for Google Collab), trains the model.  Make sure to download weights if not doing all work on Collab.
 7. `post_processing_tfrecords.py` -- post processes results of test dataset (in tfrecords format) from saved weights
 8. `explore_calculate_metrics.ipynb` -- takes in post-process results, 
 
### For exploring data

 * `explore_features.ipynb` -- look at plots of different features
 * `explore_post_procesing.ipynb` -- explore effects of post-processing steps.
 
 
## TODO

 * for `pull_check_makesense.ipynb` -- make sure there is an option when there is no model already run
 * for  `process_annotations_and_generate_features_batch.py` -- maybe this should just be annotation generation? and then `generate_features_only.py` is the follow up?
 * what file is PDFmining in?
