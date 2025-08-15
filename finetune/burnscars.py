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
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
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

def build_transform(config):
    transform_list = []
    class_path_list = []
    for item in config:
        class_path = item["class_path"]
        class_path_list.append(class_path)
        init_args = item.get("init_args", {}) or {}
        if hasattr(A, class_path):
            transform_class = getattr(A, class_path)
        elif class_path == "ToTensorV2":
            transform_class = ToTensorV2
        else:
            raise ValueError(f"Transform '{class_path}' not found.")
        
        transform = transform_class(**init_args)
        transform_list.append(transform)

    return  A.Compose(transform_list), class_path_list

def create_datamodule(data_path, dataset_bands, model_bands, means, stds, train_transform, test_transform):
    # means, stds from https://github.com/NASA-IMPACT/hls-foundation-os/blob/main/configs/burn_scars.py
    # means = [
    #     0.033349706741586264,
    #     0.05701185520536176,
    #     0.05889748132001316,
    #     0.2323245113436119,
    #     0.1972854853760658,
    #     0.11944914225186566,
    # ]
    # stds = [
    #     0.02269135568823774,
    #     0.026807560223070237,
    #     0.04004109844362779,
    #     0.07791732423672691,
    #     0.08708738838140137,
    #     0.07241979477437814,
    # ]
    
    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size = 8,
        num_workers = 0,
        num_classes = 2,

        # Define dataset paths
        train_data_root = os.path.join(data_path, "data/"),
        train_label_data_root = os.path.join(data_path, "data/"),
        val_data_root = os.path.join(data_path, "data/"),
        val_label_data_root = os.path.join(data_path, "data/"),
        test_data_root = os.path.join(data_path, "data/"),
        test_label_data_root = os.path.join(data_path, "data/"),

        # Define splits, img and label
        train_split = os.path.join(data_path, "splits/train.txt"),
        val_split = os.path.join(data_path, "splits/val.txt"),
        test_split = os.path.join(data_path, "splits/test.txt"),
        img_grep = "*_merged.tif",
        label_grep = "*.mask.tif",

        # Transformation
        means = means[0:len(model_bands)],
        stds = stds[0:len(model_bands)],
        train_transform = train_transform,
        val_transform = test_transform,
        test_transform = test_transform,

        # Bands selection
        dataset_bands = dataset_bands,
        output_bands = model_bands,

        no_data_replace = 0,
        no_label_replace = -1,
        rgb_indices=[2,1,0]
    )

    return datamodule

def create_model(decoder, loss, class_weights, bands):
    if decoder == "UNet":
        MODEL_ARGS = {
            "backbone": "prithvi_eo_v2_300",
            "decoder": "UNetDecoder",
            "num_classes": 2,
            "backbone_bands": bands,
            "backbone_pretrained": True,
            "backbone_num_frames": 1,
            "decoder_channels": [512, 256, 128, 64],
            "head_dropout": 0.1,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"},
                {"name": "LearnedInterpolateToPyramidal"}
            ]
        }
    elif decoder == "UperNet":
        MODEL_ARGS = {
            "backbone": "prithvi_eo_v2_300",
            "decoder": "UperNetDecoder",
            "num_classes": 2,
            "backbone_bands": bands,
            "backbone_pretrained": True,
            "backbone_num_frames": 1,
            "decoder_channels": 512,
            "head_dropout": 0.1,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"}
            ]
        }
    elif decoder == "Segformer":
        MODEL_ARGS = {
            "backbone": "prithvi_eo_v2_300",
            "decoder": "smp_Segformer",
            "num_classes": 2,
            "backbone_bands": bands,
            "backbone_pretrained": True,
            "backbone_num_frames": 1,
            "decoder_channels": 512,
            "head_dropout": 0.1,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"}
            ]
        }

    task = SemanticSegmentationTask(
        model_args = MODEL_ARGS,
        model_factory = "EncoderDecoderFactory",
        loss = loss,
        lr = 1e-4,
        optimizer = "AdamW",
        optimizer_hparams = {"weight_decay": 0.01},
        scheduler = "ReduceLROnPlateau",
        scheduler_hparams = {"factor": 0.5, "patience": 4},
        ignore_index = -1,
        freeze_backbone = True,
        freeze_decoder = False,
        plot_on_val = True,
        class_weights = class_weights,
        class_names = ["Not burned", "Burn scar"],
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
    DATASET_BANDS = config["data"]["dataset_bands"]
    MODEL_BANDS = config["data"]["model_bands"]
    MEANS = config["data"]["means"]
    STDS = config["data"]["stds"]
    TRAIN_TRANSFORM = config["data"]["train_transform"]
    TEST_TRANSFORM = config["data"]["test_transform"]

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
    train_transform, train_classpath_list = build_transform(TRAIN_TRANSFORM)
    test_transform, test_classpath_list = build_transform(TEST_TRANSFORM)
    logging.info(f"Using seed: {SEED_EVERYTHING}")
    logging.info(f"Using loss: {LOSS}")
    logging.info(f"Using decoder: {DECODER}")
    logging.info(f"Using class weights: {CLASS_WEIGHTS}")
    logging.info(f"Using bands: {MODEL_BANDS}")
    logging.info(f"Using means: {MEANS}")
    logging.info(f"Using stds: {STDS}")
    logging.info(f"Using train transform: {train_classpath_list}")
    logging.info(f"Using test transform: {test_classpath_list}")

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
    datamodule = create_datamodule(data_path=DATA_PATH, dataset_bands=DATASET_BANDS, model_bands=MODEL_BANDS,
                                   means=MEANS, stds=STDS, train_transform=train_transform, test_transform=test_transform)
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
    task = create_model(decoder=DECODER, loss=LOSS, class_weights=CLASS_WEIGHTS, bands=MODEL_BANDS)

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
    best_model = SemanticSegmentationTask.load_from_checkpoint(
        ckpt_path,
        model_factory = task.hparams.model_factory,
        model_args = task.hparams.model_args,
    )
    test_loader = datamodule.test_dataloader()
    with torch.no_grad():
        batch = next(iter(test_loader))
        images = datamodule.aug(batch)
        images = batch["image"].to(best_model.device)
        masks = batch["mask"].numpy()
        outputs = best_model(images)
        preds = torch.argmax(outputs.output, dim=1).cpu().numpy()
    for i in range(batch_size):
        sample = {key: batch[key][i] for key in batch}
        sample["prediction"] = preds[i]
        sample["image"] = sample["image"].cpu()
        sample["mask"] = sample["mask"].cpu()
        fig = test_dataset.plot(sample)
        plt.close(fig)
        fig.savefig(os.path.join(PREDICTION_PATH, f"predict_{i}.png"))
    logging.info(f"Completed prediction on test set: {batch_size} images")