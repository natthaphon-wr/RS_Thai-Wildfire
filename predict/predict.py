import argparse
import logging
import os
import gc
import yaml
import csv
import warnings
import numpy as np
import pandas as pd
import rasterio
from glob import glob
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

import torch
from torch import Tensor

from terratorch.datasets import HLSBands
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
from terratorch.cli_tools import LightningInferenceModel

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_datamodule(data_path, batchsize, means, stds, rgb_only):
    hls_bands = [
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ]
    test_transform = A.Compose([ToTensorV2()])
    model_bands = hls_bands[0:3] if rgb_only else hls_bands
    
    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size = batchsize,
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
        predict_output_bands = model_bands,
    
        no_data_replace = 0,
        no_label_replace = -1,
        rgb_indices = [2,1,0]
    )
    
    datamodule.setup("predict")
    predict_set = datamodule.predict_dataset

    return datamodule, predict_set

def plot_msi(sample: dict[str, Tensor], suptitle: str | None = None):
    
    def select_img(image, color_indices):
        image = image.take(color_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)
        return image

    num_classes = 2
    rgb_idx = [2,1,0]
    infared_idx = [3,2,1]   # NIR_NARROW, RED, GREEN
    swir_idx = [5,3,2]      # SWIR_2, NIR_NARROW, RED
    agri_idx = [4,3,0]      # SWIR_1, NIR_NARROW, BLUE

    # Image
    image = sample["image"]
    if isinstance(image, Tensor):
        image = image.numpy()
    rgb_img = select_img(image, rgb_idx)
    if image.shape[0]==6: # MSI
        infared_img = select_img(image, infared_idx)
        swir_img = select_img(image, swir_idx)
        agri_img = select_img(image, agri_idx)

    # Cloud and shadow mask
    colors = ['black', 'gray', 'white']
    cmap = ListedColormap(colors)
    norm_cloud = BoundaryNorm(
        boundaries = [-1.5, -0.5, 0.5, 1.5], 
        ncolors = len(colors)
    )
    cloud_mask = sample["mask"]
    if isinstance(cloud_mask, Tensor):
        cloud_mask = cloud_mask.numpy()

    # Prediction mask
    showing_predictions = "prediction" in sample
    if showing_predictions:
        prediction_mask = sample["prediction"]
        if isinstance(prediction_mask, Tensor):
            prediction_mask = prediction_mask.numpy()
    prediction = prediction_mask if showing_predictions else None
    norm_predict = mpl.colors.Normalize(vmin=0, vmax=num_classes - 1)

    # Plotting
    num_images = 7 if image.shape[0]==6 else 4
    fig, ax = plt.subplots(1, num_images, figsize=(15, 5), layout="compressed")
    axes_visibility = "off"

    ax[0].axis(axes_visibility)
    ax[0].title.set_text("Cloud and Shadow Mask")
    ax[0].imshow(cloud_mask, cmap=cmap, norm=norm_cloud)

    ax[1].axis(axes_visibility)
    ax[1].title.set_text("RGB Image")
    ax[1].imshow(rgb_img)

    if image.shape[0]==6:
        ax[2].axis(axes_visibility)
        ax[2].title.set_text("Infared Image")
        ax[2].imshow(infared_img)

        ax[3].axis(axes_visibility)
        ax[3].title.set_text("SWIR Image")
        ax[3].imshow(swir_img)

        ax[4].axis(axes_visibility)
        ax[4].title.set_text("Agriculture Image")
        ax[4].imshow(agri_img)

    pred_idx_fig = 5 if image.shape[0]==6 else 2
    ax[pred_idx_fig].axis(axes_visibility)
    ax[pred_idx_fig].title.set_text("Predicted Mask")
    ax[pred_idx_fig].imshow(prediction, cmap="jet", norm=norm_predict)

    cmap = plt.get_cmap("jet")
    legend_data = []
    for i, _ in enumerate(range(num_classes)):
        class_name = str(i)
        data = [i, cmap(norm_predict(i)), class_name]
        legend_data.append(data)
    handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
    labels = [n for k, c, n in legend_data]
    ax[pred_idx_fig+1].legend(handles, labels, loc="center")
    ax[pred_idx_fig+1].axis("off")

    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.close(fig)
    
    return fig

def prediction(model, datamodule, output_path, visual_tiles):
    predict_loader = datamodule.predict_dataloader()
    plot_dir = os.path.join(output_path, "example_plot")
    pred_dir = os.path.join(output_path, "pred_masks")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    best_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(predict_loader, desc="Predicting")):
            images = batch["image"].to(model.device)
            outputs = model(images)
            preds = torch.argmax(outputs.output, dim=1).cpu().numpy()
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                sample = {key: batch[key][i] for key in batch}
                sample["prediction"] = preds[i]
                sample["image"] = sample["image"].cpu()
                casename = os.path.splitext(os.path.basename(sample["filename"]))[0]
                np.save(os.path.join(pred_dir, f"{casename}.npy"), sample["prediction"]) # save prediction mask

                # save plot from visual_tiles
                casename_split = casename.split("_")
                tile_id = f"{casename_split[0]}_{casename_split[1]}_{casename_split[2]}"
                if visual_tiles["tile"].str.contains(tile_id).any():
                    fig = plot_msi(sample=sample)
                    fig.savefig(os.path.join(plot_dir, f"{casename}.png"))
                    del fig  

                del sample

            del batch, images, outputs, preds
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for HLS Thai prediction")
    parser.add_argument("--data_path", type=str, help="Data directory for prediction")
    parser.add_argument("--model_path", type=str, help="Model directory containing config and checkpoint")
    parser.add_argument("--output_path", type=str, help="Output directory")
    parser.add_argument("--rgb_only", action="store_true", help="Flag to use only RGB")

    args = parser.parse_args()
    DATA_PATH = args.data_path
    MODEL_PATH = args.model_path
    output_dir = args.output_path
    rgb_only = args.rgb_only

    dt_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_PATH = os.path.join(output_dir, f"{dt_now}")
    MODEL_CONFIG_PATH = os.path.join(MODEL_PATH, "log/lightning_logs/version_0/hparams.yaml")
    MODEL_CKPT_PATH = glob(os.path.join(MODEL_PATH, "checkpoint/best-epoch=*.ckpt"))[0]
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    logging.info(f"Completed create output directory: {OUTPUT_PATH}")

    # Create datamodule
    means = []
    stds = []
    with open(os.path.join(DATA_PATH, "stats.txt"), "r") as f:
        for line in f:
            if line.startswith("Means:"):
                numbers = line.strip().split('[')[1].split(']')[0].split()
                means = [float(n) for n in numbers]
            elif line.startswith("Stds:"):
                numbers = line.strip().split('[')[1].split(']')[0].split()
                stds = [float(n) for n in numbers]
    if rgb_only:
        means = means[0:3]
        stds = stds[0:3]
        logging.info("Use only RGB bands")
    else:
        logging.info("Use MSI")
    logging.info(f"Means: {means}")
    logging.info(f"Stds: {stds}")
    datamodule, predict_set = create_datamodule(os.path.join(DATA_PATH, "data"), 
                                                batchsize=8, 
                                                means=means, 
                                                stds=stds, 
                                                rgb_only=rgb_only)
    logging.info(f"No. images: {len(predict_set)}")

    # Load model and task
    with open(MODEL_CONFIG_PATH, "r") as file:
        config_model = yaml.safe_load(file)

    model_args = config_model["model_args"]
    model_factory = config_model["model_factory"]
    lr = config_model["lr"]
    loss = config_model["loss"]
    class_weights = config_model["class_weights"]
    optimizer = config_model["optimizer"]
    optimizer_hparams = config_model["optimizer_hparams"]
    scheduler = config_model["scheduler"]
    scheduler_hparams = config_model["scheduler_hparams"]
    freeze_backbone = config_model["freeze_backbone"]
    freeze_decoder = config_model["freeze_decoder"]
    class_names = config_model["class_names"]

    task = SemanticSegmentationTask(
        model_args = model_args,
        model_factory = model_factory,
        loss = loss,
        lr = lr,
        optimizer = optimizer,
        optimizer_hparams = optimizer_hparams,
        scheduler = scheduler,
        scheduler_hparams = scheduler_hparams,
        freeze_backbone = freeze_backbone,
        freeze_decoder = freeze_decoder,
        class_weights = class_weights,
        class_names = class_names,
    )

    best_model = SemanticSegmentationTask.load_from_checkpoint(
        MODEL_CKPT_PATH,
        model_factory = task.hparams.model_factory,
        model_args = task.hparams.model_args,
    )
    logging.info("Completed load model.")

    # Perform prediction
    visual_tiles = pd.read_csv(os.path.join(DATA_PATH, "visual_tiles.csv"))
    logging.info(f"Visualize {len(visual_tiles)} tiles id, so total {len(visual_tiles)*22} images")
    prediction(best_model, datamodule, OUTPUT_PATH, visual_tiles)
    logging.info("Completed prediction.")
