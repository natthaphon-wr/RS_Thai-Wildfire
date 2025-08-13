import argparse
import logging
import os
import yaml
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

from terratorch.datasets import HLSBands
from terratorch.datamodules import BurnIntensityNonGeoDataModule
from terratorch.datasets.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels
from terratorch.tasks import SemanticSegmentationTask

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_datamodule(data_path):
    datamodule = BurnIntensityNonGeoDataModule(
        data_root = data_path,
        batch_size = 8,
        num_workers = 0,
        
        train_transform = [
            FlattenTemporalIntoChannels(),   # Required for temporal data
            A.D4(),  
            ToTensorV2(),
            UnflattenTemporalFromChannels(n_timesteps=3), # Required for temporal data
        ],
        val_transform = None,   # Using ToTensor() by default
        test_transform = None,  
        use_full_data = True,
        no_label_replace = -1
    )

    return datamodule

def create_model(decoder, loss, class_weights, bands):
    if decoder == "UNet":
        MODEL_ARGS = {
            "backbone": "prithvi_eo_v2_300",
            "decoder": "UNetDecoder",
            "num_classes": 5,
            "backbone_bands": bands,
            "backbone_pretrained": True,
            "backbone_num_frames": 3,  
            "decoder_channels": [512, 256, 128, 64],
            "head_dropout": 0.2,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage", "effective_time_dim": 3},
                {"name": "LearnedInterpolateToPyramidal"}
            ]
        }
    elif decoder == "UperNet":
        MODEL_ARGS = {
            "backbone": "prithvi_eo_v2_300",
            "decoder": "UperNetDecoder",
            "num_classes": 5,
            "backbone_bands": bands,
            "backbone_pretrained": True,
            "backbone_num_frames": 3,
            "decoder_channels": 512,
            "head_dropout": 0.2,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage", "effective_time_dim": 3},
            ]
        }
    elif decoder == "Segformer":
        MODEL_ARGS = {
            "backbone": "prithvi_eo_v2_300",
            "decoder": "smp_Segformer",
            "num_classes": 5,
            "backbone_bands": bands,
            "backbone_pretrained": True,
            "backbone_num_frames": 3,
            "decoder_channels": 512,
            "head_dropout": 0.2,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage", "effective_time_dim": 3}
            ]
        }

    task = SemanticSegmentationTask(
        model_args = MODEL_ARGS,
        model_factory = "EncoderDecoderFactory",
        loss = loss,
        optimizer = "AdamW",
        optimizer_hparams = {"weight_decay": 0.1},
        lr = 1e-5,
        scheduler = "StepLR",
        scheduler_hparams = {"step_size": 3, "gamma": 0.9},
        ignore_index = -1,
        freeze_backbone = True,
        freeze_decoder = False,
        plot_on_val = True,
        class_names = ["No Burn", "Unburned to Very Low", "Low Severity", "Moderate Severity", "High Severity"],
        class_weights = class_weights
    )

    return task

def create_trainer(checkpoint_path, log_path):
    checkpoint_callback = ModelCheckpoint(
        dirpath = checkpoint_path,
        mode = "max",
        monitor = "val/Multiclass_Jaccard_Index",
        filename = "best-{epoch:02d}",
        save_weights_only = False,
        save_last = True,
    )
    early_stopping_callback = EarlyStopping(monitor="val/loss", patience = 15)
    tb_logger = TensorBoardLogger(save_dir = log_path)
    csv_logger = CSVLogger(save_dir = log_path)

    trainer = Trainer(
        precision="16-mixed",
        callbacks=[
            RichProgressBar(),
            checkpoint_callback,
            early_stopping_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger = [tb_logger, csv_logger],
        max_epochs = 100,
        default_root_dir = log_path,
        log_every_n_steps = 10,
        check_val_every_n_epoch = 1
    )

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A basic Python script with arguments.")
    parser.add_argument("--config_path", type=str, help="YAML configuration")

    args = parser.parse_args()
    config_path = args.config_path

    # Read YAML configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    SEED_EVERYTHING = config["seed_everything"]
    DATA_PATH = config["path"]["data_path"]
    output_path = config["path"]["output_path"]
    LOSS = config["model"]["loss"]
    CLASS_WEIGHTS = config["model"]["class_weights"]
    DECODER = config["model"]["decoder"]

    allow_loss = {"ce", "dice"}
    allow_decoder = {"UNet", "UperNet", "Segformer"}
    if LOSS not in allow_loss:
        raise ValueError(
            f"Invalid loss '{LOSS}'. Must be one of: {', '.join(allow_loss)}"
        )
    if DECODER not in allow_decoder:
        raise ValueError(
            f"Invalid decoder '{DECODER}'. Must be one of: {', '.join(allow_decoder)}"
        )
    BANDS = [
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ]
    logging.info(f"Using seed: {SEED_EVERYTHING}")
    logging.info(f"Using loss: {LOSS}")
    logging.info(f"Using decoder: {DECODER}")
    logging.info(f"Using class weights: {CLASS_WEIGHTS}")

    # Define and create directory
    dt_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_PATH = os.path.join(output_path, f"{dt_now}")
    LOG_PATH = os.path.join(OUTPUT_PATH, "log")
    CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoint")
    PERFORMANCE_PATH = os.path.join(OUTPUT_PATH, "performance")
    PREDICTION_PATH = os.path.join(OUTPUT_PATH, "prediction")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(PERFORMANCE_PATH, exist_ok=True)
    os.makedirs(PREDICTION_PATH, exist_ok=True)
    logging.info(f"Completed create output directory: {OUTPUT_PATH}")

    # Datamodule
    datamodule = create_datamodule(DATA_PATH)
    batch_size = datamodule.batch_size
    datamodule.setup("fit")
    datamodule.setup("test")
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset
    logging.info(f"No. of train set : {len(train_dataset)}")
    logging.info(f"No. of val set : {len(val_dataset)}")
    logging.info(f"No. of test set : {len(test_dataset)}")

    # Model and task
    task = create_model(decoder=DECODER, loss=LOSS, class_weights=CLASS_WEIGHTS, bands=BANDS)

    # Trainer
    trainer = create_trainer(checkpoint_path=CHECKPOINT_PATH, log_path=LOG_PATH)

    # Training and testing
    trainer.fit(model=task, datamodule=datamodule)
    res_train = trainer.callback_metrics
    ckpt_path = trainer.checkpoint_callback.best_model_path
    res_test = trainer.test(dataloaders=datamodule, ckpt_path=ckpt_path)
    dict_result = {**res_train, **res_test[0]}
    dict_result = {k: v.item() if torch.is_tensor(v) and v.numel() == 1 else v for k, v in dict_result.items()}
    with open(os.path.join(PERFORMANCE_PATH, "performance.json"), "w") as json_file:
        json.dump(dict_result, json_file, indent=4)
    logging.info("Completed train and test model.")

    # Prediction on test set
    # At terratorch==1.0.2, BurnIntensityNonGeoDataModule is only support train, val split. (predict_dataset = val_dataset) 
    #   So,I manual change for using test-set for prediction-set"
    preds = trainer.predict(task, datamodule=datamodule, ckpt_path=ckpt_path)
    data_loader = trainer.predict_dataloaders
    batch = next(iter(data_loader))
    for i in range(batch_size):
        sample = {key: batch[key][i] for key in batch}
        sample["prediction"] = preds[0][0][0][i].cpu().numpy()
        fig = datamodule.predict_dataset.plot(sample)
        plt.close(fig)
        fig.savefig(os.path.join(PREDICTION_PATH, f"predict_{i}.png"))
    logging.info(f"Completed prediction on test set: {batch_size} images")