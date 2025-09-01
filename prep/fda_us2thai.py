import argparse
import logging
import warnings
import os
import csv
import shutil
import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm
from glob import glob

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def low_freq_replace(amp_src, amp_trg, beta=0.002):
    h, w = amp_src.shape
    b = int(np.floor(min(h, w) * beta))

    # Center square region (low frequency)
    c_h, c_w = h // 2, w // 2
    h1, h2 = c_h - b, c_h + b
    w1, w2 = c_w - b, c_w + b
    patch_center_trg = amp_trg[h1:h2, w1:w2]

    if np.any(np.isnan(patch_center_trg)): # Have nan value
        return amp_src

    amp_src[h1:h2, w1:w2] = patch_center_trg
    return amp_src

def FDA(src_img, trg_img, beta=0.002):
    src_img = src_img.astype(np.float32)
    trg_img = trg_img.astype(np.float32)
    src_in_trg  = np.zeros_like(src_img)

    for b in range(src_img.shape[0]):
        # Fourier Transform each band
        src_fft = np.fft.fft2(src_img[b])
        trg_fft = np.fft.fft2(trg_img[b])

        # Shift zero-freq to center
        src_amp_shift = np.fft.fftshift(np.abs(src_fft))
        src_phase = np.angle(src_fft)
        trg_amp_shift = np.fft.fftshift(np.abs(trg_fft))

        # Replace source low freq to target's 
        src_amp_shift = low_freq_replace(src_amp_shift, trg_amp_shift, beta=beta)

        # Inverse shift
        src_amp_new = np.fft.ifftshift(src_amp_shift)

        # Inverse fourier transform
        fft_new = src_amp_new * np.exp(1j * src_phase)
        src_in_trg[b] = np.fft.ifft2(fft_new).real

    # Replace nan to 0
    src_in_trg = np.nan_to_num(src_in_trg, nan=0.0)
    
    return src_in_trg

def fda_augmentation(source_dir, ref_dir, output_dir, beta=0.002, ratio=0.5):
    source_data_dir = os.path.join(source_dir, "data")
    source_split_dir = os.path.join(source_dir, "splits")
    output_data_dir = os.path.join(output_dir, "data")
    output_split_dir = os.path.join(output_dir, "splits")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

    # Perform FDA and save images
    src_images = [f for f in os.listdir(source_data_dir) if f.endswith("_merged.tif")]
    ref_images = [f for f in os.listdir(ref_dir) if f.endswith(".tif")]
    csv_path = os.path.join(output_dir, "fda_check.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "is_fda"])

        for src_img_name in tqdm(src_images, desc="FDA augment"):
            src_img_path = os.path.join(source_data_dir, src_img_name)
            with rasterio.open(src_img_path) as src:
                src_img = src.read()
                src_profile = src.profile

            # Keep original is default
            is_fda = 0
            final_img = src_img

            # Apply FDA
            if np.random.rand() < ratio: # Uniform random in [0,1)
                ref_name = np.random.choice(ref_images)
                ref_path = os.path.join(ref_dir, ref_name)
                with rasterio.open(ref_path) as src:
                    ref_img = src.read()
                final_img = FDA(src_img, ref_img, beta=beta).astype(np.float32)
                is_fda = 1

            # Save image
            out_path = os.path.join(output_data_dir, src_img_name)
            with rasterio.open(out_path, "w", **src_profile) as dst:
                dst.write(final_img)

            # Log FDA on each images
            writer.writerow([src_img_name, is_fda])

    # Copy splits and labels
    shutil.copytree(source_split_dir, output_split_dir)
    for fname in tqdm(os.listdir(source_data_dir), desc="Copy mask"):
        if fname.endswith("mask.tif"):
            src_path = os.path.join(source_data_dir, fname)
            dst_path = os.path.join(output_data_dir, fname)
            shutil.copy(src_path, dst_path)

def compute_mean_std(folder_path, desc=None):
    image_paths = glob(os.path.join(folder_path, "*merged.tif"))

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
    parser = argparse.ArgumentParser(description="Arguments for FDA (Fourier Domain Adpatation) on MSI")
    parser.add_argument("--data_path", type=str, help="Source data directory")
    parser.add_argument("--ref_path", type=str, help="Reference data directory")
    parser.add_argument("--output_path", type=str, help="Output data directory")
    parser.add_argument("--beta", type=float, help="Beta parameter for FDA")
    parser.add_argument("--ratio", type=float, help="Ratio of images to perform FDA")

    args = parser.parse_args()
    DATA_PATH = args.data_path
    REF_PATH = args.ref_path
    OUTPUT_PATH = args.output_path
    BETA = args.beta
    RATIO = args.ratio

    # Perform FDA Augmentation
    fda_augmentation(
        source_dir = DATA_PATH, 
        ref_dir = REF_PATH, 
        output_dir = OUTPUT_PATH, 
        beta = BETA,
        ratio = RATIO
    )
    logging.info("Finish performing FDA augmentation")

    # Check is_fda
    fda_df = pd.read_csv(os.path.join(OUTPUT_PATH, "fda_check.csv"))
    logging.info(f"No. of augmented images: {sum(fda_df["is_fda"]==1)}")
    logging.info(f"No. of original images: {sum(fda_df["is_fda"]==0)}")

    # Calculate means, stds for new dataset
    mean, std = compute_mean_std(folder_path=os.path.join(DATA_PATH, "data"))
    lines = [f"Means: {mean}", f"Stds: {std}"]
    with open(os.path.join(OUTPUT_PATH, "stats.txt"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    logging.info(f"Mean: {mean}")
    logging.info(f"Std: {std}")
