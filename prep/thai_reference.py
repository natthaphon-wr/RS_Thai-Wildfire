import argparse
import logging
import warnings
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob

from terratorch.datasets import HLSBands
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

def get_cloud_ratio(datamodule):
    df = pd.DataFrame(columns=["path", "cloud_ratio"])
    for i in tqdm(range(len(datamodule)), desc="Get cloud ratio"):
        cloud_ratio = ((datamodule[i]["mask"] == 1)).sum().item()/(512*512)
        df_tmp = pd.DataFrame([{"path": datamodule[i]["filename"], "cloud_ratio": cloud_ratio}])
        df = pd.concat([df, df_tmp], ignore_index=True)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for sampling Thai as reference for FDA, Histogram Matching")
    parser.add_argument("--data_path", type=str, help="Source data directory")
    parser.add_argument("--target_path", type=str, help="Target data directory to save data")
    parser.add_argument("--n", type=int, help="Number of images to get (on each pos/neg)")

    args = parser.parse_args()
    DATA_PATH = args.data_path
    DATA_PATH_TARGET = args.target_path
    N_SAMPLE = args.n

    DATA_PATH_POS = os.path.join(DATA_PATH, "positive")
    DATA_PATH_NEG = os.path.join(DATA_PATH, "negative")
    os.makedirs(DATA_PATH_TARGET, exist_ok=True)

    datamodule_pos = create_datamodule(DATA_PATH_POS)
    datamodule_neg = create_datamodule(DATA_PATH_NEG)
    pos_df = get_cloud_ratio(datamodule_pos)
    neg_df = get_cloud_ratio(datamodule_neg)
    pos_clear = pos_df[pos_df["cloud_ratio"]==0]
    neg_clear = neg_df[neg_df["cloud_ratio"]==0]
    logging.info("No. of clear images (no cloud)")
    logging.info(f"Positive group: {len(pos_clear)}")
    logging.info(f"Negative group: {len(neg_clear)}")

    pos_sample = pos_clear.sample(N_SAMPLE)
    neg_sample = neg_clear.sample(N_SAMPLE)
    for path in tqdm(pos_sample["path"], desc="Copy positive samples"):
        shutil.copy(path, DATA_PATH_TARGET)
    for path in tqdm(neg_sample["path"], desc="Copy negative samples"):
        shutil.copy(path, DATA_PATH_TARGET)
    logging.info(f"Finish create Thai referenced images with total {N_SAMPLE*2} images.")