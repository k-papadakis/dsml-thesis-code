from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from torch.utils.tensorboard.writer import SummaryWriter

from thesis.dataloading import load_electricity
from thesis.metrics import METRICS

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")

ROOT_DIR = Path("output", "electricity", "nbeats")

# data loading
data, freq = load_electricity("./datasets/electricity/")
data = (
    data.reset_index()
    .reset_index()
    .rename(columns={"index": "time_idx"})
    .set_index(["time_idx", "date"])
    .rename_axis("series", axis="columns")
    .stack()
    .rename("value")  # type: ignore
    .reset_index()
)

# slicing configuration
horizon = pd.Timedelta(1, "day")

output_length = horizon // freq
input_length = 7 * output_length
validation_cutoff = data["time_idx"].max() - output_length
training_cutoff = validation_cutoff - 21 * output_length

assert pd.DataFrame.equals(
    data[(data["series"] == "MT_001") & (data["time_idx"] <= validation_cutoff)],
    data[(data["series"] == "MT_001") & (data["date"] <= data["date"].max() - horizon)],
)

print(
    f"{input_length = }\n{output_length = }\n{validation_cutoff = }\n{training_cutoff = }\n{data['time_idx'].max() = }"
)

# datasets and dataloaders
train = TimeSeriesDataSet(
    data[data["time_idx"] <= training_cutoff],
    time_idx="time_idx",
    target="value",
    group_ids=["series"],
    # only unknown variable is "value" - and N-Beats can also not take any additional variables
    time_varying_unknown_reals=["value"],
    max_encoder_length=input_length,
    max_prediction_length=output_length,
)
val = TimeSeriesDataSet.from_dataset(
    train,
    data[data["time_idx"] <= validation_cutoff],
    min_prediction_idx=training_cutoff + 1,
)
test = TimeSeriesDataSet.from_dataset(
    train,
    data,
    # min_prediction_idx=validation_cutoff + 1,
    predict=True,
)

print(f"{len(train) = }\n{len(val) = }\n{len(test) = }")

batch_size = 64
train_dataloader = train.to_dataloader(
    train=True,
    batch_size=batch_size,
    num_workers=2,
)
val_dataloader = val.to_dataloader(
    train=False,
    batch_size=batch_size,
    num_workers=2,
)
test_dataloader = test.to_dataloader(
    train=False,
    batch_size=batch_size,
    num_workers=0,
)

# model
# TODO: No percentage loss in Traffic
model = NBeats.from_dataset(
    train,
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
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
_ = trainer.test(ckpt_path="best", dataloaders=test_dataloader)

# Further Logging

# load best
best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore
best_model = NBeats.load_from_checkpoint(best_model_path)

# predict
out = best_model.predict(
    test_dataloader, mode="raw", return_x=True, return_y=True, return_index=True
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
