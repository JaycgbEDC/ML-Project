import torch
import json
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import OilTemparatureForecasting


train_data_path = "data/processed_train_set.csv"
val_data_path = "data/processed_validation_set.csv"
test_data_path = "data/processed_test_set.csv"
log_dir = "models/oil_temparature_forecasting_logs"
model_dir = "models/oil_temparature_forecasting_models"
output_json_path = "models/trained_config.json"

features = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - 192 + 1  # the number of time series
    
    def __getitem__(self, idx):
        start_index = idx
        lable_index = idx + 96
        end_index = lable_index + 96
        history = np.array(self.data[start_index:lable_index][features + ["OT"]])
        trg = self.data[lable_index:end_index]
        trg_in = np.array(trg[features + ["OT_lag_1"]])
        trg_out = np.array(trg["OT"])
        return torch.tensor(history, dtype=torch.float), torch.tensor(trg_in, dtype=torch.float), torch.tensor(trg_out, dtype=torch.float)


def train(
    epochs: int = 2000
):
    data_train = pd.read_csv(train_data_path)
    data_val = pd.read_csv(val_data_path)
    train_data = Dataset(data_train)
    val_data = Dataset(data_val)
    
    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        num_workers=10,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=32,
        num_workers=10,
        shuffle=False,
    )

    model = OilTemparatureForecasting(
        n_encoder_inputs=len(features) + 1,
        n_decoder_inputs=len(features) + 1,
        lr=1e-5,
        dropout=0.1,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[0, 1],
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

    return output_json


if __name__ == '__main__':
    train()