import argparse
import logging
import warnings
import os
import glob
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

from terratorch.datasets import HLSBands
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def split_img(img_path, output_folder, new_size, forest_tiles):
    file_names = []
    with rasterio.open(img_path) as src:
        meta = src.meta.copy()
        width, height = src.width, src.height
        new_width, new_height = new_size
        transform = src.transform
        crs = src.crs

        for i in range(0, width-new_width, new_width):
            for j in range(0, height-new_height, new_height):
                row_str = str(j).zfill(4)
                col_str = str(i).zfill(4)
                fname = os.path.splitext(os.path.basename(img_path))[0]
                small_patch = fname.split("_")[0] + f"_{row_str}_{col_str}"

                if (forest_tiles==small_patch).any() :
                    w = min(new_width, width - i)
                    h = min(new_height, height - j)
                    window = Window(i, j, w, h)

                    new_x, new_y = transform * (i, j)
                    new_transform = from_origin(new_x, new_y, transform.a, -transform.e)
                    meta.update({
                        "width": w,
                        "height": h,
                        "transform": new_transform,
                        "crs": crs
                    })
    
                    new_img = src.read(window=window)
                    nan_ratio = np.isnan(new_img).sum()/(new_img.shape[1]*new_img.shape[2]*6)
                    nonzero_ratio = np.count_nonzero(new_img)/(new_img.shape[1]*new_img.shape[2]*6)
                    if nan_ratio <= 0.05 and nonzero_ratio >= 0.95:
                        new_fname = small_patch + "_" + fname.split("_")[1] + ".tif"
                        file_names.append(new_fname)
                        new_filepath = os.path.join(output_folder, new_fname)
                        with rasterio.open(new_filepath, "w", **meta) as dst:
                            dst.write(src.read(window=window))
            
    return file_names

def create_datamodule(data_path):
    hls_bands = [
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ]
    test_transform = A.Compose([ToTensorV2()])
    means = [0.02, 0.04, 0.03, 0.09, 0.08, 0.05] # approximate
    stds = [0.06, 0.10, 0.10, 0.20, 0.20, 0.13] # approximate
    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size = 8,
        num_workers = 2,
        num_classes = 2,

        train_data_root = None,
        val_data_root = None,
        test_data_root = None,
        predict_data_root = data_path,
        img_grep = "*.tif",

        means = means,
        stds = stds,
        test_transform = test_transform,

        predict_dataset_bands = hls_bands,
        predict_output_bands = hls_bands,

        no_data_replace = 0,
        no_label_replace = -1,
        rgb_indices = [2,1,0]
    )
    datamodule.setup("predict")
    datamodule_predict = datamodule.predict_dataset
    return datamodule_predict

def filter_cloud(datamodule, img_size):
    count = 0
    for i in tqdm(range(len(datamodule)), desc="Filter cloud ratio"):
        cloud_ratio = ((datamodule[i]["mask"] == 1)).sum().item()/(img_size*img_size)
        if cloud_ratio > 0:
            count += 1
            os.remove(datamodule[i]["filename"])
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for preparation of Thai reference set for FDA")
    parser.add_argument("--raw_data_path", type=str, help="Raw data directory")
    parser.add_argument("--prep_data_path", type=str, help="Preprocess data directory")
    parser.add_argument("--img_size", type=int, help="Preprocess image size")
    parser.add_argument("--n", type=int, help="Number of images to get (on each pos/neg)")
    
    args = parser.parse_args()
    DATA_PATH_RAW = args.raw_data_path
    DATA_PATH_PREPROCESS = args.prep_data_path
    IMG_SIZE = args.img_size
    N_SAMPLE = args.n

    DATA_PATH_RAW_POSITVE = os.path.join(DATA_PATH_RAW, "positive")
    DATA_PATH_RAW_NEGATIVE = os.path.join(DATA_PATH_RAW, "negative")
    DATA_PATH_TMP_POSITIVE = os.path.join(DATA_PATH_PREPROCESS, "tmp_pos")
    DATA_PATH_TMP_NEGATIVE = os.path.join(DATA_PATH_PREPROCESS, "tmp_neg")
    os.makedirs(DATA_PATH_PREPROCESS, exist_ok=True)
    os.makedirs(DATA_PATH_TMP_POSITIVE, exist_ok=True)
    os.makedirs(DATA_PATH_TMP_NEGATIVE, exist_ok=True)

    # Read forest tile csv
    forest_tiles = pd.read_csv(os.path.join(DATA_PATH_RAW, "forest_tiles.csv"))
    forest_tiles["Row_str"] = forest_tiles["Row"].apply(lambda x: f"{x:04d}")
    forest_tiles["Column_str"] = forest_tiles["Column"].apply(lambda x: f"{x:04d}")
    forest_tiles["small_tile"] = "T" + forest_tiles["MGRS_Tile"] + "_" + \
                                 forest_tiles["Row_str"] + "_" + forest_tiles["Column_str"]
    logging.info(f"Number of small tiles that are forest: {len(forest_tiles)}")
    logging.info("Example:")
    logging.info(forest_tiles.head())

    # Split images
    img_size = (IMG_SIZE, IMG_SIZE)
    pos_raw_filelist = os.listdir(DATA_PATH_RAW_POSITVE)
    neg_raw_filelist = os.listdir(DATA_PATH_RAW_NEGATIVE)
    pos_file_name = []
    for fname in tqdm(pos_raw_filelist, desc="Split positive"):
        file_path = os.path.join(DATA_PATH_RAW_POSITVE, fname)
        file_names = split_img(img_path=file_path, 
                               output_folder=DATA_PATH_TMP_POSITIVE, 
                               new_size=img_size, 
                               forest_tiles=forest_tiles["small_tile"])
        pos_file_name = pos_file_name + file_names
    logging.info(f"Completed split positve set from {len(pos_raw_filelist)} to {len(pos_file_name)}")
    neg_file_name = []
    for fname in tqdm(neg_raw_filelist, desc="Split negative"):
        file_path = os.path.join(DATA_PATH_RAW_NEGATIVE, fname)
        file_names = split_img(img_path=file_path, 
                               output_folder=DATA_PATH_TMP_NEGATIVE, 
                               new_size=img_size, 
                               forest_tiles=forest_tiles["small_tile"])
        neg_file_name = neg_file_name + file_names
    logging.info(f"Completed split negative set from {len(neg_raw_filelist)} to {len(neg_file_name)}")
    
    # Filter out cloud
    datamodule_pos = create_datamodule(DATA_PATH_TMP_POSITIVE)
    logging.info(f"Length of positive datamodule: {len(datamodule_pos)}") 
    count_del_pos = filter_cloud(datamodule=datamodule_pos, img_size=IMG_SIZE)
    logging.info(f"Remove {count_del_pos} images from cloud filter")
    datamodule_neg = create_datamodule(DATA_PATH_TMP_NEGATIVE)
    logging.info(f"Length of negative datamodule: {len(datamodule_neg)}") 
    count_del_neg = filter_cloud(datamodule=datamodule_neg, img_size=IMG_SIZE)
    logging.info(f"Remove {count_del_neg} images from cloud filter")

    # Sampling 
    pos_prep_filelist = os.listdir(DATA_PATH_TMP_POSITIVE)
    neg_prep_filelist = os.listdir(DATA_PATH_TMP_NEGATIVE)
    pos_sampling = random.sample(pos_prep_filelist, N_SAMPLE)
    neg_sampling = random.sample(neg_prep_filelist, N_SAMPLE) 
    for fname in tqdm(pos_sampling, desc="Move positive samples"):
        shutil.move(os.path.join(DATA_PATH_TMP_POSITIVE, fname), DATA_PATH_PREPROCESS)
    for fname in tqdm(neg_sampling, desc="Move negative samples"):
        shutil.move(os.path.join(DATA_PATH_TMP_NEGATIVE, fname), DATA_PATH_PREPROCESS)
    shutil.rmtree(DATA_PATH_TMP_POSITIVE)
    shutil.rmtree(DATA_PATH_TMP_NEGATIVE)
    logging.info("Complete move sampling file and delete tmp folders")