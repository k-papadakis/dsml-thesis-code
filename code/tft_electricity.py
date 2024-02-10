from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    QuantileLoss,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from torch.utils.tensorboard.writer import SummaryWriter

from thesis.dataloading import SeriesDataModule
from thesis.metrics import METRICS

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")

ROOT_DIR = Path("output", "electricity", "tft")

# data loading
datamodule = SeriesDataModule(
    name="electricity",
    path="./datasets/electricity/",
    multivariate=False,
    with_covariates=True,
    transform="auto",
    batch_size=128,
)
datamodule.setup("test")
test_dataset = datamodule.test

# model
model = TemporalFusionTransformer.from_dataset(
    test_dataset,
    hidden_size=160,
    hidden_continuous_size=160,
    dropout=0.1,
    lstm_layers=1,
    attention_head_size=4,
    output_size=3,
    learning_rate=1e-3,
    log_interval=300,
    optimizer="Ranger",
    loss=QuantileLoss([0.1, 0.5, 0.9]),
    logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]),
)

# trainer
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, mode="min", verbose=False
)
checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False)
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[early_stop_callback, checkpoint_callback],
    gradient_clip_val=0.01,
    gradient_clip_algorithm="norm",
    default_root_dir=ROOT_DIR,
)

# Further Logging

# fit
trainer.fit(model, datamodule=datamodule)
_ = trainer.test(ckpt_path="best", datamodule=datamodule)

# load best
best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore
best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# predict
out = best_model.predict(
    test_dataset,
    mode="raw",
    return_x=True,
    return_y=True,
    return_index=True,
)

performance = {
    metric_fn.__name__: {
        name: metric_fn(y_true.cpu().numpy(), y_pred.cpu().numpy()).item()
        for name, y_true, y_pred in zip(
            out.index["series"],
            out.y[0],
            out.output.prediction.mean(-1),
        )
    }
    for metric_fn in METRICS
}

# TODO: Find a way to log quantile losses.

pd.DataFrame(performance).to_csv(Path(trainer.log_dir, "performance.csv"))  # type: ignore

# Tensorboard
summary_writer: SummaryWriter = trainer.logger.experiment  # type: ignore

# plot preds
for i, name in out.index["series"].items():
    fig = best_model.plot_prediction(
        out.x,
        out.output,
        idx=i,
    )
    summary_writer.add_figure(f"prediction/{name}", fig)

predictions_vs_actuals = best_model.calculate_prediction_actual_by_variable(
    out.x, out.output.prediction
)
figs = best_model.plot_prediction_actual_by_variable(predictions_vs_actuals)

for name, fig in figs.items():  # type: ignore
    summary_writer.add_figure(f"prediction_actual_by_variable/{name}", fig)
