import argparse
import logging
import warnings
import os
import glob
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

def split_img(img_path, pred_dir, new_size, forest_tiles, prediction_tiles):
    count = 0
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
                        if (prediction_tiles==small_patch).any():
                            new_fname = small_patch + "_" + fname.split("_")[1] + ".tif"
                            new_filepath = os.path.join(pred_dir, new_fname)
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

def get_cloud_ratio(datamodule, img_size):
    df = pd.DataFrame(columns=["file_name", "cloud_ratio"])
    for i in tqdm(range(len(datamodule)), desc="Get cloud ratio"):
        cloud_ratio = ((datamodule[i]["mask"] == 1)).sum().item()/(img_size*img_size)
        df_tmp = pd.DataFrame([{"file_name": os.path.basename(datamodule[i]["filename"]), "cloud_ratio": cloud_ratio}])
        df = pd.concat([df, df_tmp], ignore_index=True)
    
    return df

def get_means_stds(path):
    def compute_stats(data_path):
        image_paths = glob.glob(os.path.join(data_path, "*.tif"))
        # Find percentiles
        band_pixels = [[] for _ in range(6)]
        for img_path in image_paths:
            with rasterio.open(img_path) as src:
                img = src.read() 
                for band in range(img.shape[0]):
                    band_data = img[band]
                    valid_pixels = band_data[band_data > 0]
                    if valid_pixels.size > 0:
                        band_pixels[band].append(valid_pixels)
        p2 = np.zeros(6)
        p98 = np.zeros(6)
        for band in range(6):
            all_vals = np.concatenate(band_pixels[band])
            p2[band] = np.percentile(all_vals, 2)
            p98[band] = np.percentile(all_vals, 98)

        # Find stats
        sum_ = None
        sum_sq = None
        count = 0
        for img_path in tqdm(image_paths, "Compute stats"):
            with rasterio.open(img_path) as src:
                img = src.read() 
                if sum_ is None:
                    sum_ = np.zeros(img.shape[0], dtype=np.float64)
                    sum_sq = np.zeros(img.shape[0], dtype=np.float64)
                for band in range(img.shape[0]):
                    band_data = img[band]
                    valid_pixels = band_data[band_data > 0]
                    valid_pixels = valid_pixels[(valid_pixels >= p2[band]) & (valid_pixels <= p98[band])]
                    if valid_pixels.size > 0:
                        sum_[band] += valid_pixels.sum()
                        sum_sq[band] += np.square(valid_pixels).sum()
                        count += len(valid_pixels)

        mean = sum_ / count
        std = np.sqrt((sum_sq / count) - np.square(mean))
        return mean, std

    mean, std = compute_stats(data_path=os.path.join(path, "data"))
    lines = [f"Means: {mean}", f"Stds: {std}"]
    with open(os.path.join(path, "stats.txt"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for preparation of HLS Thai prediction data")
    parser.add_argument("--raw_data_path", type=str, help="Directory of raw HLS Thai data")
    parser.add_argument("--pred_data_path", type=str, help="Directory of prediction dataset")
    parser.add_argument("--img_size", type=int, default=512, help="Preprocess image size")

    args = parser.parse_args()
    RAW_DATA_PATH = args.raw_data_path
    PRED_DATA_PATH = args.pred_data_path
    IMG_SIZE = args.img_size
    os.makedirs(PRED_DATA_PATH, exist_ok=True)
    os.makedirs(os.path.join(PRED_DATA_PATH, "data"), exist_ok=True)

    # Read forest tiles csv
    forest_tiles = pd.read_csv(os.path.join(RAW_DATA_PATH, "forest_tiles.csv"))
    forest_tiles["Row_str"] = forest_tiles["Row"].apply(lambda x: f"{x:04d}")
    forest_tiles["Column_str"] = forest_tiles["Column"].apply(lambda x: f"{x:04d}")
    forest_tiles["small_tile"] = "T" + forest_tiles["MGRS_Tile"] + "_" + \
                                 forest_tiles["Row_str"] + "_" + forest_tiles["Column_str"]
    logging.info(f"Number of small tiles that are forest: {len(forest_tiles)}")
    logging.info("Example:")
    logging.info(forest_tiles.head())

    # Read prediction tiles csv
    predict_tiles = pd.read_csv(os.path.join(RAW_DATA_PATH, "prediction_tiles.csv"))
    logging.info(f"Number of prediction tiles: {len(predict_tiles)}")
    logging.info("Example:")
    logging.info(predict_tiles.head())

    # Split images into prediction and reference set
    img_size = (IMG_SIZE, IMG_SIZE)
    files_list = glob.glob(os.path.join(RAW_DATA_PATH, "*.tif"))
    count = 0
    for file_path in tqdm(files_list, desc="Split raw images"):
        count += split_img(img_path=file_path,
                           pred_dir=os.path.join(PRED_DATA_PATH, "data"),
                           new_size=img_size,
                           forest_tiles=forest_tiles["small_tile"],
                           prediction_tiles=predict_tiles["tile"])
    logging.info(f"Completed split images from {len(files_list)} to {count}")
    
    # Get cloud ratio
    datamodule_pred = create_datamodule(os.path.join(PRED_DATA_PATH, "data"))
    logging.info(f"Length of prediction datamodule: {len(datamodule_pred)}")
    cloud_pred_df = get_cloud_ratio(datamodule=datamodule_pred, img_size=IMG_SIZE)
    cloud_pred_df.to_csv(os.path.join(PRED_DATA_PATH, "cloud_ratio.csv"))
    logging.info("Saving cloud ratio of each images to csv")

    # Find means and stds
    means_pred, stds_pred = get_means_stds(path=PRED_DATA_PATH)
    logging.info("Saved means/stds of prediction set")
    logging.info(f"Means: {means_pred}")
    logging.info(f"Stds: {stds_pred}")