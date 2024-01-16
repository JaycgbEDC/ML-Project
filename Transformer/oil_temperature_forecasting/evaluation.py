import json
from typing import Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from oil_temperature_forecasting.model import OilTemparatureForecasting
from oil_temperature_forecasting.training import Dataset


val_data_path = "data/processed_validation_set.csv"
trained_json_path = "models/trained_config.json"
eval_json_path = "data/eval.json"
data_for_visualization_path = "data/visualization.json"

features = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]


def mse_loss(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return np.sum((true - pred)**2)


def mae_loss(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return np.sum(np.abs(true - pred))


def evaluate_regression(true, pred):
    """
    eval mae + smape
    :param true:
    :param pred:
    :return:
    """

    return {"mse": mse_loss(true, pred), "mae": mean_absolute_error(true, pred)}


def evaluate(
    trained_json_path: str,
    eval_json_path: str,
    horizon_size: int = 96,
    data_for_visualization_path: Optional[str] = None,
):
    """
    Evaluates the model on the last 8 labeled weeks of the data.
    Compares the model to a simple baseline : prediction the last known value
    :param data_csv_path:
    :param feature_target_names_path:
    :param trained_json_path:
    :param eval_json_path:
    :param horizon_size:
    :param data_for_visualization_path:
    :return:
    """
    with open(trained_json_path) as f:
        model_json = json.load(f)

    model_path = model_json["best_model_path"]

    target = "OT"

    data_val = pd.read_csv(val_data_path)
    val_data = Dataset(data_val)

    model = OilTemparatureForecasting(
        n_encoder_inputs=len(features) + 1,
        n_decoder_inputs=len(features) + 1,
        lr=1e-4,
        dropout=0.5,
    )
    model.load_state_dict(torch.load(model_path)["state_dict"])

    model.eval()

    gt = []
    baseline_last_known_values = []
    neural_predictions = []

    data_for_visualization = []

    for i in tqdm(range(100), desc='evaluation_tqdm'):
        time_series_data = {"history": [], "ground_truth": [], "prediction": []}

        start_index = i
        lable_index = i + 96
        end_index = lable_index + 96
        src, trg = data_val[start_index:lable_index][[features + ["OT"]]], data_val[lable_index:end_index]

        time_series_data["history"] = src[target].tolist()[-96:]
        time_series_data["ground_truth"] = trg[target].tolist()

        last_known_value = src[target].values[-1]

        trg.loc[:, "last_known_value"] = last_known_value

        gt += trg[target].tolist()
        baseline_last_known_values += trg["last_known_value"].tolist()

        src, trg_in, _ = val_data[i]

        src, trg_in = src.unsqueeze(0), trg_in.unsqueeze(0)

        with torch.no_grad():
            prediction = model((src, trg_in[:, :1, :]))
            for j in range(1, horizon_size):
                last_prediction = prediction[0, -1]
                trg_in[:, j, -1] = last_prediction
                prediction = model((src, trg_in[:, : (j + 1), :]))

            trg[target + "_predicted"] = (prediction.squeeze().numpy()).tolist()

            neural_predictions += trg[target + "_predicted"].tolist()

            time_series_data["prediction"] = trg[target + "_predicted"].tolist()

        data_for_visualization.append(time_series_data)

    baseline_eval = evaluate_regression(gt, baseline_last_known_values)
    model_eval = evaluate_regression(gt, neural_predictions)

    eval_dict = {
        "Baseline_MAE": baseline_eval["mae"],
        "Baseline_MSE": baseline_eval["mse"],
        "Model_MAE": model_eval["mae"],
        "Model_MSE": model_eval["mse"],
    }

    if eval_json_path is not None:
        with open(eval_json_path, "w") as f:
            json.dump(eval_dict, f, indent=4)

    if data_for_visualization_path is not None:
        with open(data_for_visualization_path, "w") as f:
            json.dump(data_for_visualization, f, indent=4)

    for k, v in eval_dict.items():
        print(k, v)

    return eval_dict


if __name__ == "__main__":
    evaluate(
        trained_json_path=trained_json_path,
        eval_json_path=eval_json_path,
        data_for_visualization_path=data_for_visualization_path,
    )
