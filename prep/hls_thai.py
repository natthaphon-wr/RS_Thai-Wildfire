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

def split_img(img_path, output_folder, new_size, forest_tiles):
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
                        new_fname = small_patch + "_" + fname.split("_")[1] + ".tif"
                        new_filepath = os.path.join(output_folder, new_fname)
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

def filter_cloud(datamodule, threshold):
    count = 0
    for i in tqdm(range(len(datamodule)), desc="Filter cloud ratio"):
        cloud_ratio = ((datamodule[i]["mask"] == 1)).sum().item()/(512*512)
        if cloud_ratio > threshold:
            count += 1
            os.remove(datamodule[i]["filename"])
    return count

def compute_mean_std_hls(folder_path, desc):
    image_paths = glob.glob(os.path.join(folder_path, "*.tif"))

    # Find percentiles
    band_pixels = [[] for _ in range(6)]
    for img_path in tqdm(image_paths, desc="Find percentiles"):
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
    for img_path in tqdm(image_paths, desc=desc):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for preparation of HLS Thai data")
    parser.add_argument("--raw_data_path", type=str, help="Raw data directory")
    parser.add_argument("--prep_data_path", type=str, help="Preprocess data directory")
    parser.add_argument("--img_size", type=int, help="Preprocess image size")
    parser.add_argument("--cloud_threshold", type=float, help="Cloud threshold for filter images")

    args = parser.parse_args()
    DATA_PATH_RAW = args.raw_data_path
    DATA_PATH_PREPROCESS = args.prep_data_path
    IMG_SIZE = args.img_size
    CLOUD_THRESHOLD = args.cloud_threshold
    os.makedirs(DATA_PATH_PREPROCESS, exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH_PREPROCESS, "data"), exist_ok=True)

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
    files_list = glob.glob(os.path.join(DATA_PATH_RAW, "*.tif"))
    count = 0
    for file_path in tqdm(files_list, desc="Split raw images"):
        count += split_img(file_path, os.path.join(DATA_PATH_PREPROCESS, "data"), img_size, forest_tiles["small_tile"])
    logging.info(f"Completed split images from {len(files_list)} to {count}")
    
    # Filter out cloud
    datamodule = create_datamodule(os.path.join(DATA_PATH_PREPROCESS, "data"))
    logging.info(f"Length of datamodule: {len(datamodule)}")
    count_del = filter_cloud(datamodule, CLOUD_THRESHOLD)
    logging.info(f"Remove {count_del} images from cloud filter")

    # Find means and stds
    mean, std = compute_mean_std_hls(folder_path=os.path.join(DATA_PATH_PREPROCESS, "data"), desc="Find means/stds")
    lines = [f"Means: {mean}", f"Stds: {std}"]
    with open(os.path.join(DATA_PATH_PREPROCESS, "stats.txt"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    logging.info(f"Means: {mean}")
    logging.info(f"Stds: {std}")

    # Get tiles id that have all periods
    file_list = glob.glob(os.path.join(DATA_PATH_PREPROCESS, "data", "*.tif"))
    filename_list = [os.path.basename(path) for path in file_list]
    rows = []
    for fname in filename_list:
        fname_split = fname.split("_")
        tile = f"{fname_split[0]}_{fname_split[1]}_{fname_split[2]}"
        year = fname_split[3][0:4]
        month_date = fname_split[3][4:8]
        rows.append({"tile":tile, "year":year, "month_date": month_date})
    df = pd.DataFrame(rows)
    df_filter = df.groupby("tile")["month_date"].count().reset_index(name="row_count")
    df_filter = df_filter[df_filter["row_count"]==22]
    selected_tile_id = df_filter["tile"]
    selected_tile_id.to_csv(os.path.join(DATA_PATH_PREPROCESS, "visual_tiles.csv"), index=False)
    logging.info(f"There are {len(df_filter)} tiles id that have all periods")
    logging.info("Save it to csv")