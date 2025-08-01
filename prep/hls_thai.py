import argparse
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

from terratorch.datasets import HLSBands
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def split_img(img_path, output_folder, new_size):
    count = 0
    with rasterio.open(img_path) as src:
        meta = src.meta.copy()
        width, height = src.width, src.height
        new_width, new_height = new_size
        transform = src.transform
        crs = src.crs

        for i in range(0, width-new_width, new_width):
            for j in range(0, height-new_height, new_height):
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

                row_str = str(j).zfill(4)
                col_str = str(i).zfill(4)
                new_fname = os.path.splitext(os.path.basename(img_path))[0] + f"_{row_str}_{col_str}" + ".tif"
                new_filepath = os.path.join(output_folder, new_fname)
                new_img = src.read(window=window)
                nan_ratio = np.isnan(new_img).sum()/(new_img.shape[1]*new_img.shape[2]*6)
                nonzero_ratio = np.count_nonzero(new_img)/(new_img.shape[1]*new_img.shape[2]*6)
                if nan_ratio <= 0.05 and nonzero_ratio >= 0.95:
                    with rasterio.open(new_filepath, "w", **meta) as dst:
                        dst.write(src.read(window=window))
                    count += 1
            
    return count

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
    means = [0.02, 0.04, 0.04, 0.09, 0.09, 0.06] # approx
    stds = [0.08, 0.12, 0.13, 0.21, 0.23, 0.17] # approx

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

def filter_cloud(datamodule):
    count = 0
    for i in tqdm(range(len(datamodule)), desc="Filter cloud ratio"):
        cloud_ratio = ((datamodule[i]["mask"] == 1)).sum().item()/(512*512)
        if cloud_ratio > 0.1:
            count += 1
            os.remove(datamodule[i]["filename"])
    return count

def delete_surplus(file_list, new_index, data_path, desc):
    pos_del = set(file_list) - set(new_index)
    for filename in tqdm(pos_del, desc=desc):
        filepath = os.path.join(data_path, filename)
        try:
            os.remove(filepath)
        except FileNotFoundError:
            print(f"Not found: {filepath}")
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")

    return pos_del

def compute_mean_std_hls(folder_path, desc):
    image_paths = glob(os.path.join(folder_path, "*.tif"))
    sum_ = None
    sum_sq = None
    count = 0

    for img_path in tqdm(image_paths, desc=desc):
        with rasterio.open(img_path) as src:
            img = src.read() 
            mask = img > 0
            valid = img[mask]
            if valid.size == 0:
                continue  # skip if no valid pixels
            if sum_ is None:
                sum_ = np.zeros(img.shape[0], dtype=np.float64)
                sum_sq = np.zeros(img.shape[0], dtype=np.float64)
            for band in range(img.shape[0]):
                band_data = img[band]
                valid_pixels = band_data[band_data > 0]
                sum_[band] += valid_pixels.sum()
                sum_sq[band] += np.square(valid_pixels).sum()
                count += len(valid_pixels)

    mean = sum_ / count
    std = np.sqrt((sum_sq / count) - np.square(mean))

    return mean, std

def combine_mean_std(mean_1, std_1, n_1, mean_2, std_2, n_2):
    combined_n = n_1 + n_2
    mean = (n_1 * mean_1 + n_2 * mean_2) / combined_n
    std = np.sqrt(
        (n_1 * (std_1**2 + (mean_1 - mean)**2) + n_2 * (std_2**2 + (mean_2 - mean)**2)) / combined_n
    )
    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A basic Python script with arguments.")
    parser.add_argument("--raw_data_path", type=str, help="Raw data directory")
    parser.add_argument("--prep_data_path", type=str, help="Preprocess data directory")

    args = parser.parse_args()
    DATA_PATH_RAW = args.raw_data_path
    DATA_PATH_PREPROCESS = args.prep_data_path

    # Define and create directory
    DATA_PATH_RAW_POSITVE = os.path.join(DATA_PATH_RAW, "positive")
    DATA_PATH_RAW_NEGATIVE = os.path.join(DATA_PATH_RAW, "negative")
    DATA_PATH_PREPROCESS_POSITIVE = os.path.join(DATA_PATH_PREPROCESS, "positive")
    DATA_PATH_PREPROCESS_NEGATIVE = os.path.join(DATA_PATH_PREPROCESS, "negative")

    # Read raw index
    index_raw = pd.read_csv(os.path.join(DATA_PATH_RAW, "index.csv"))
    logging.info(f"Index have {len(index_raw)} pairs")

    # Split images
    img_size = (512, 512)
    pos_file_list = os.listdir(DATA_PATH_RAW_POSITVE)
    neg_file_list = os.listdir(DATA_PATH_RAW_NEGATIVE)
    pos_count = 0
    for fname in tqdm(pos_file_list, desc="Split positive"):
        file_path = os.path.join(DATA_PATH_RAW_POSITVE, fname)
        pos_count += split_img(file_path, DATA_PATH_PREPROCESS_POSITIVE, img_size)
    logging.info(f"Completed split positve set from {len(pos_file_list)} to {pos_count}")
    neg_count = 0
    for fname in tqdm(neg_file_list, desc="Split negative"):
        file_path = os.path.join(DATA_PATH_RAW_NEGATIVE, fname)
        neg_count += split_img(file_path, DATA_PATH_PREPROCESS_NEGATIVE, img_size)
    logging.info(f"Completed split negative set from {len(neg_file_list)} to {neg_count}")
    
    # Filter out cloud
    datamodule_pos = create_datamodule(DATA_PATH_PREPROCESS_POSITIVE)
    datamodule_neg = create_datamodule(DATA_PATH_PREPROCESS_NEGATIVE)
    logging.info(f"Length of positive datamodule: {len(datamodule_pos)}")
    logging.info(f"Length of negative datamodule: {len(datamodule_neg)}")
    count_del_pos = filter_cloud(datamodule_pos)
    count_del_neg = filter_cloud(datamodule_neg)
    logging.info(f"Remove positive {count_del_pos} images for filter cloud")
    logging.info(f"Remove negative {count_del_neg} images for filter cloud")

    # Create preprocess index
    width, height = (3660, 3660)
    new_width, new_height = (512,  512)
    suffix_list = []
    for i in range(0, width-new_width, new_width):
        for j in range(0, height-new_height, new_height):
            row_str = str(j).zfill(4)
            col_str = str(i).zfill(4)
            suffix_list.append(f"_{row_str}_{col_str}")
    index_preprocess = pd.DataFrame(
        [(burn_id + suffix + ".tif", unburn_id + suffix + ".tif")
        for burn_id, unburn_id in zip(index_raw['Burned ID'], index_raw['Unburned ID'])
        for suffix in suffix_list],
        columns=['positive', 'negative']
    )
    pos_files = glob(f"{DATA_PATH_PREPROCESS_POSITIVE}/*.tif")
    pos_files = [os.path.basename(path) for path in pos_files]
    neg_files = glob(f"{DATA_PATH_PREPROCESS_NEGATIVE}/*.tif")
    neg_files = [os.path.basename(path) for path in neg_files]
    logging.info(f"Total splited positive images: {len(pos_files)}")
    logging.info(f"Total splited negative images: {len(neg_files)}")
    idx_in_pos = index_preprocess[index_preprocess["positive"].isin(pos_files)].index
    idx_in_neg = index_preprocess[index_preprocess["negative"].isin(neg_files)].index
    idx_intesection = set(idx_in_pos) & set(idx_in_neg)
    logging.info(f"There are {len(idx_intesection)} pairs that are intersection.")
    index_preprocess_filter = index_preprocess[index_preprocess.index.isin(idx_intesection)]
    index_preprocess_filter = index_preprocess_filter.drop_duplicates(subset="negative", keep="first")
    index_preprocess_filter.to_csv(os.path.join(DATA_PATH_PREPROCESS, "index.csv"), index=False)
    logging.info(f"There are {len(index_preprocess_filter)} pairs after drop duplicate.")
    logging.info("Completed create preprocess index.")

    # Delete surplus images
    pos_del = delete_surplus(file_list=pos_files, new_index=index_preprocess_filter["positive"], 
                            data_path=DATA_PATH_PREPROCESS_POSITIVE, desc="Delete positve surplus")
    neg_del = delete_surplus(file_list=neg_files, new_index=index_preprocess_filter["negative"], 
                            data_path=DATA_PATH_PREPROCESS_NEGATIVE, desc="Delete negative surplus")

    # Find means and stds
    mean_pos, std_pos = compute_mean_std_hls(folder_path=DATA_PATH_PREPROCESS_POSITIVE, desc="Find stats for positive")
    mean_neg, std_neg = compute_mean_std_hls(folder_path=DATA_PATH_PREPROCESS_NEGATIVE, desc="Find stats for negative")
    mean, std = combine_mean_std(mean_1=mean_pos, std_1=std_pos, n_1=len(index_preprocess_filter), 
                                mean_2=mean_neg, std_2=std_neg, n_2=len(index_preprocess_filter))
    lines = [f"Means: {mean}", f"Stds: {std}"]
    with open(os.path.join(DATA_PATH_PREPROCESS, "stats.txt"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    logging.info(f"Means: {mean}")
    logging.info(f"Stds: {std}")