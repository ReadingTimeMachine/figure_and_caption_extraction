# figure_and_caption_extraction

## Order

 1. change paths in config.py for system
 2. `ocr_and_image_processing_batch.py` -- runs the OCR and pdfmining to get raw data from pages
 3. `pull_check_makesense.ipynb` -- uses a current model to "guess" boxes, and prepares them to check with MakeSense.ai
 4. `process_annotations_and_generate_features_batch.py` -- process annotations after downloading (as a csv file) from MakeSense.ai, generate features as well if you want
 
 
## TODO

 * for `pull_check_makesense.ipynb` -- make sure there is an option when there is no model already run
 * for  `process_annotations_and_generate_features_batch.py` -- maybe this should just be annotation generation? and then `generate_features_only.py` is the follow up?
