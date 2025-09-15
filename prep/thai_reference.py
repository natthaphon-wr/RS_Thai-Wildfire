import argparse
import logging
import warnings
import os
import shutil
import pandas as pd
import glob
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for create Thai reference for FDA")
    parser.add_argument("--prediction_path", type=str, help="Data directory for Thai prediction dataset")
    parser.add_argument("--reference_path", type=str, help="Data directory for reference dataset")
    parser.add_argument("--n_month", type=int, help="Number of sample images in each month")
    
    args = parser.parse_args()
    PREDICTION_PATH = args.prediction_path
    REFERENCE_PATH = args.reference_path
    n_month = args.n_month
    os.makedirs(REFERENCE_PATH, exist_ok=True)

    # Selected prediction tiles
    predict_tiles = pd.read_csv(os.path.join(PREDICTION_PATH, "visual_tiles.csv"))
    logging.info("Prediction tiles:")
    logging.info(predict_tiles)

    # Select reference images
    file_list = glob.glob(os.path.join(PREDICTION_PATH, "data", "*.tif"))
    filename_list = [os.path.basename(path) for path in file_list]
    rows = []
    for fname in filename_list:
        fname_split = fname.split("_")
        filename = fname
        tile = f"{fname_split[0]}_{fname_split[1]}_{fname_split[2]}"
        year = fname_split[3][0:4]
        month = fname_split[3][4:6]
        rows.append({"filename":filename, "tile":tile, "year":year, "month": month})
    df = pd.DataFrame(rows)
    df_exclude = df[~df["tile"].isin(predict_tiles["tile"])]
    logging.info(f"No. of total prediction: {len(df)}")
    logging.info(f"No. of excluded prediction: {len(df_exclude)}")
    reference = df.groupby('month', group_keys=False).apply(lambda x: x.sample(n=10))
    logging.info(f"Sampling {n_month} per month, so total {len(reference)} reference images")

    # Copy files
    for file_name in tqdm(reference["filename"]):
        src = os.path.join(PREDICTION_PATH, "data", file_name)
        dst = os.path.join(REFERENCE_PATH, file_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            logging.info(f"File not found: {src}")
