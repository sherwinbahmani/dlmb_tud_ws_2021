# Setup dataset

- Download dataset and put the files in a directory ../dataset/raw:

    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

- Unzip the zip files in the directory

- Run this file to setup the dataset for training, validation and testing

    ```
    python datasets/split_dataset.py --dataset-raw PATH_TO_DATASET_RAW
    ```
   
Path required to a directory with DATASET_RAW ../dataset/raw:
  - ../dataset/raw/
    - HAM10000_images_part_1
    - HAM10000_images_part_2
    - HAM10000_segmentations_lesion_tschandl
    - HAM10000_metadata.csv

Creates a new subdirectory ../dataset/split:
  - ../dataset/split/
    - train
        - img
            - xxx.png
        - gt
            - xxx.png
        - labels.csv
    - val
        - img
            - xxx.png
        - gt
            - xxx.png
        - labels.csv
    - test
        - img
            - xxx.png
        - gt
            - xxx.png
        - labels.csv