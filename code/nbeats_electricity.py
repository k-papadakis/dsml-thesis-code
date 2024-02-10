from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import NBeats
from torch.utils.tensorboard.writer import SummaryWriter

from thesis.dataloading import SeriesDataModule
from thesis.metrics import METRICS

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")

ROOT_DIR = Path("output", "electricity", "nbeats")

datamodule = SeriesDataModule(
    name="electricity",
    path="./datasets/electricity/",
    multivariate=False,
    with_covariates=False,
    transform="auto",
    batch_size=64,
)
datamodule.setup("test")
test_dataset = datamodule.test

# model
model = NBeats.from_dataset(
    test_dataset,
    expansion_coefficient_lengths=[3, 2],
    widths=[256, 2048],
    learning_rate=1e-4,
    log_interval=300,
)

# trainer
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, mode="min", verbose=False
)
checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False)
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[early_stop_callback, checkpoint_callback],
    gradient_clip_val=0.1,
    gradient_clip_algorithm="norm",
    default_root_dir=ROOT_DIR,
)

# fit
trainer.fit(model, datamodule=datamodule)
_ = trainer.test(ckpt_path="best", datamodule=datamodule)

# Further Logging

# load best
best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore
best_model = NBeats.load_from_checkpoint(best_model_path)

# predict
out = best_model.predict(
    test_dataset, mode="raw", return_x=True, return_y=True, return_index=True
)

performance = {
    metric_fn.__name__: {
        name: metric_fn(y_true.cpu().numpy(), y_pred.cpu().numpy()).item()
        for name, y_true, y_pred in zip(
            out.index["series"],
            out.y[0],
            out.output.prediction,
        )
    }
    for metric_fn in METRICS
}

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

# plot interpretation
for i, name in out.index["series"].items():
    fig = best_model.plot_interpretation(out.x, out.output, idx=i)
    summary_writer.add_figure(f"interpretation/{name}", fig)
