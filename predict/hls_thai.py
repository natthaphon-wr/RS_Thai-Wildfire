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
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

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

def create_datamodule(data_path, batchsize, means, stds, use_rgb):
    hls_bands = [
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ]
    test_transform = A.Compose([ToTensorV2()])
    model_bands = hls_bands[0:3] if use_rgb else hls_bands
    
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

def plot_custom(sample: dict[str, Tensor], suptitle: str | None = None):
    
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

def calculate_area(sample: dict[str, Tensor]):
    mask = sample["mask"].numpy()
    prediction = sample["prediction"]
    filepath = sample["filename"]
    filename = os.path.basename(filepath)
    
    # Cloud, shadow mask
    is_cloud = (mask==1)
    is_shadow = (mask==-1)
    is_obscured = is_cloud | is_shadow
    is_clear = (mask==0)
    
    # Prediction mask
    is_burned = (prediction==1)
    is_unburned = (prediction==0)
    is_burned_cloud = is_burned & is_cloud
    is_burned_shadow = is_burned & is_shadow
    is_burned_obscured = is_burned & is_obscured
    is_burned_clear = is_burned & is_clear
    
    # Sum area
    area_cloud = np.sum(is_cloud)
    area_shadow = np.sum(is_shadow)
    area_obscured = np.sum(is_obscured)
    area_clear = np.sum(is_clear)
    area_burned = np.sum(is_burned)
    area_unburned = np.sum(is_unburned)
    area_burned_cloud = np.sum(is_burned_cloud)
    area_burned_shadow = np.sum(is_burned_shadow)
    area_burned_obscured  = np.sum(is_burned_obscured)
    area_burned_clear = np.sum(is_burned_clear)

    return {
        "filename": filename,
        "cloud": area_cloud, 
        "shadow": area_shadow,
        "obscured": area_obscured,
        "clear": area_clear,
        "burned": area_burned,
        "unburned": area_unburned,
        "burned_cloud": area_burned_cloud,
        "burned_shadow": area_burned_shadow,
        "burned_obscured": area_burned_obscured,
        "burned_clear": area_burned_clear
    }

def prediction(model, datamodule, output_path, visualized_sample, desc):
    predict_loader = datamodule.predict_dataloader()
    plot_dir = os.path.join(output_path, "example_plot")
    pred_dir = os.path.join(output_path, "pred_masks")
    obscured_dir = os.path.join(output_path, "obscured_masks")
    csv_path = os.path.join(output_path, "results.csv")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(obscured_dir, exist_ok=True)
    fieldnames = ["filename", "cloud", "shadow", "obscured", "clear", "burned", "unburned", 
                "burned_cloud", "burned_shadow", "burned_obscured", "burned_clear"]
    
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        best_model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(predict_loader, desc=desc)):
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
                    np.save(os.path.join(obscured_dir, f"{casename}.npy"), sample["mask"]) # save obscured mask
                    res = calculate_area(sample=sample)
                    writer.writerow(res)

                    # save plot from visualized_samples
                    match_row = visualized_sample[visualized_sample==casename]
                    if not match_row.empty:
                        fig = plot_custom(sample=sample)
                        fig.savefig(os.path.join(plot_dir, f"{match_row.index[0]}_{casename}.png"))
                        del fig  
    
                    del sample
    
                del batch, images, outputs, preds
                gc.collect()
                torch.cuda.empty_cache()

def analysis(res_pos, res_neg, total_pixels, output_path):
    def create_boxplot(data_pos, data_neg, title):
        fig, ax = plt.subplots()
        ax.boxplot([data_pos, data_neg], labels=["Positive", "Negative"])
        ax.set_ylabel("Burned Ratio")
        ax.set_title(title)
        plt.close(fig)
        return fig

    def compare_all(res_pos, res_neg, total_pixels):
        res_pos["ratio_burned"] = res_pos["burned"] / total_pixels
        res_neg["ratio_burned"] = res_neg["burned"] / total_pixels
        fig = create_boxplot(data_pos=res_pos["ratio_burned"], 
                            data_neg=res_neg["ratio_burned"], 
                            title="All Region")
        return fig, res_pos, res_neg
    
    def cloud_vs_clear(res_pos, res_neg, total_pixels):
        res_pos["ratio_cloud"] = res_pos["cloud"] / total_pixels
        res_pos["ratio_burned_on_clear"] = (res_pos["burned_clear"]/res_pos["clear"]).fillna(0)
        res_pos["ratio_burned_on_cloud"] = (res_pos["burned_cloud"]/res_pos["cloud"]).fillna(0)
        res_neg["ratio_cloud"] = res_neg["cloud"] / total_pixels
        res_neg["ratio_burned_on_clear"] = (res_neg["burned_clear"]/res_neg["clear"]).fillna(0)
        res_neg["ratio_burned_on_cloud"] = (res_neg["burned_cloud"]/res_neg["cloud"]).fillna(0)
        fig_clear = create_boxplot(data_pos=res_pos["ratio_burned_on_clear"], 
                                data_neg=res_neg["ratio_burned_on_clear"], 
                                title="Clear Region")
        fig_cloud = create_boxplot(data_pos=res_pos["ratio_burned_on_cloud"],
                                data_neg=res_neg["ratio_burned_on_cloud"],
                                title="Cloud Region")
        return fig_clear, fig_cloud, res_pos, res_neg
    
    def corr_cloud_burned(res_pos, res_neg):
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
        ax[0].scatter(x=res_pos["ratio_cloud"], y=res_pos["ratio_burned"])
        ax[0].set_title("Positive")
        ax[1].scatter(x=res_neg["ratio_cloud"], y=res_neg["ratio_burned"])
        ax[1].set_title("Negative")
        for ax in ax:
            ax.set_xlabel("Cloud Ratio")
            ax.set_ylabel("Burn Ratio")
        plt.suptitle("Scatter plot between cloud ratio and burned ratio")
        plt.tight_layout()
        plt.close(fig)
        return fig
    
    fig_all, res_pos, res_neg = compare_all(res_pos, res_neg, total_pixels)
    fig_clear, fig_cloud, res_pos, res_neg = cloud_vs_clear(res_pos, res_neg, total_pixels)
    fig_scatter = corr_cloud_burned(res_pos, res_neg)
    fig_all.savefig(os.path.join(output_path, "boxplot_all.png"))
    fig_clear.savefig(os.path.join(output_path, "boxplot_clear.png"))
    fig_cloud.savefig(os.path.join(output_path, "boxplot_cloud.png"))
    fig_scatter.savefig(os.path.join(output_path, "scatter_corr.png"))

    return res_pos, res_neg

def sum_res(df):
    cols = ["ratio_burned", "ratio_cloud", "ratio_burned_on_clear", "ratio_burned_on_cloud"]
    summary_df = pd.DataFrame({
        "mean": df[cols].mean(),
        "median": df[cols].median(),
        "min": df[cols].min(),
        "max": df[cols].max(),
        "std": df[cols].std()
    })
    summary_df = summary_df.reset_index().rename(columns={"index": "measures"})
    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A basic Python script with arguments.")
    parser.add_argument("--data_path", type=str, help="Data directory for prediction")
    parser.add_argument("--model_path", type=str, help="Model directory containing config and checkpoint")
    parser.add_argument("--output_path", type=str, help="Output directory")
    parser.add_argument("--n_sample", type=int, default=10, help="Number of pairs for visualization output")
    parser.add_argument("--use_rgb", action="store_true", help="Flag to use only RGB")

    args = parser.parse_args()
    DATA_PATH = args.data_path
    MODEL_PATH = args.model_path
    output_dir = args.output_path
    n_sample = args.n_sample
    use_rgb = args.use_rgb

    dt_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_POS_PATH = os.path.join(DATA_PATH, "positive")
    DATA_NEG_PATH = os.path.join(DATA_PATH, "negative")
    OUTPUT_PATH = os.path.join(output_dir, f"{dt_now}")
    OUTPUT_POS_PATH = os.path.join(OUTPUT_PATH, "positive")
    OUTPUT_NEG_PATH = os.path.join(OUTPUT_PATH, "negative")
    OUTPUT_ANALYSE_PATH = os.path.join(OUTPUT_PATH, "analysis")
    MODEL_CONFIG_PATH = os.path.join(MODEL_PATH, "log/lightning_logs/version_0/hparams.yaml")
    MODEL_CKPT_PATH = glob(os.path.join(MODEL_PATH, "checkpoint/best-epoch=*.ckpt"))[0]
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(OUTPUT_POS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_NEG_PATH, exist_ok=True)
    os.makedirs(OUTPUT_ANALYSE_PATH, exist_ok=True)

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
    if use_rgb:
        means = means[0:3]
        stds = stds[0:3]
        logging.info("Use only RGB bands")
    else:
        logging.info("Use MSI")
    logging.info(f"Means: {means}")
    logging.info(f"Stds: {stds}")
    datamodule_pos, predict_set_pos = create_datamodule(DATA_POS_PATH, batchsize=8, means=means, stds=stds, use_rgb=use_rgb)
    datamodule_neg, predict_set_neg = create_datamodule(DATA_NEG_PATH, batchsize=8, means=means, stds=stds, use_rgb=use_rgb)
    logging.info(f"No. image of positive group: {len(predict_set_pos)}")
    logging.info(f"No. image of negative group: {len(predict_set_neg)}")

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

    # Sample pairs for visualization to compare
    index = pd.read_csv(os.path.join(DATA_PATH, "index.csv"))
    samples = index.sample(n_sample)
    samples["positive"] = samples["positive"].str.rsplit(".", n=1).str[0]
    samples["negative"] = samples["negative"].str.rsplit(".", n=1).str[0]
    logging.info(f"Sample pairs for visualization: {n_sample}")
    logging.info("Example:")
    logging.info(samples.head())

    # Perform prediction
    prediction(model=best_model, datamodule=datamodule_pos, output_path=OUTPUT_POS_PATH, 
            visualized_sample=samples["positive"], desc="Predicting positve")
    prediction(model=best_model, datamodule=datamodule_neg, output_path=OUTPUT_NEG_PATH, 
            visualized_sample=samples["negative"], desc="Predicting negative")
    logging.info("Completed prediction.")

    # Analysis results
    total_pixel = 512*512
    res_pos = pd.read_csv(os.path.join(OUTPUT_POS_PATH, "results.csv"))
    res_pos.drop(columns=["shadow", "obscured", "burned_shadow", "burned_obscured"], inplace=True)
    res_neg = pd.read_csv(os.path.join(OUTPUT_NEG_PATH, "results.csv"))
    res_neg.drop(columns=["shadow", "obscured", "burned_shadow", "burned_obscured"], inplace=True)
    res_pos, res_neg = analysis(res_pos, res_neg, total_pixel, OUTPUT_ANALYSE_PATH)
    summary_pos = sum_res(res_pos)
    summary_neg = sum_res(res_neg)
    summary_pos.to_csv(os.path.join(OUTPUT_ANALYSE_PATH, "summary_pos.csv"), index=False)
    summary_neg.to_csv(os.path.join(OUTPUT_ANALYSE_PATH, "summary_neg.csv"), index=False)
    logging.info("Completed all result analysis.")
