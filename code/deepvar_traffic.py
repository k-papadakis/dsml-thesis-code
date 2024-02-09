from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import (
    DeepAR,
    MultivariateNormalDistributionLoss,
    TimeSeriesDataSet,
)
from torch.utils.tensorboard.writer import SummaryWriter

from thesis.dataloading import load_traffic
from thesis.metrics import METRICS

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")

ROOT_DIR = Path("output", "traffic", "deepvar")

# data loading
data, freq = load_traffic("./datasets/traffic/")
data = (
    data.dropna()
    .reset_index()
    .reset_index()
    .rename(columns={"index": "time_idx"})
    .set_index(["time_idx", "date"])
    .rename_axis("series", axis="columns")
    .stack()
    .rename("value")  # type: ignore
    .reset_index()
)
data["weekday"] = data["date"].dt.weekday.astype("string").astype("category")
data["hour"] = data["date"].dt.hour.astype("string").astype("category")
data["series"] = data["series"].astype("string").astype("category")

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
    time_varying_unknown_reals=["value"],
    max_encoder_length=input_length,
    max_prediction_length=output_length,
    time_varying_known_categoricals=["hour", "weekday"],
    static_categoricals=["series"],
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

batch_size = 128
train_dataloader = train.to_dataloader(
    train=True, batch_size=batch_size, num_workers=2, batch_sampler="synchronized"
)
val_dataloader = val.to_dataloader(
    train=False, batch_size=batch_size, num_workers=2, batch_sampler="synchronized"
)
test_dataloader = test.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)

# model
model = DeepAR.from_dataset(
    train,
    learning_rate=1e-2,
    hidden_size=30,
    rnn_layers=2,
    optimizer="Adam",
    loss=MultivariateNormalDistributionLoss(quantiles=[0.1, 0.5, 0.9], rank=30),
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

# load best
best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore
best_model = DeepAR.load_from_checkpoint(best_model_path)

# predict
out = best_model.predict(
    test_dataloader,
    mode="raw",
    return_x=True,
    return_y=True,
    return_index=True,
    n_samples=100,
)

# Correlation matrix of the average prediction random variable (84 predictions)
cov = (
    best_model.loss.map_x_to_distribution(
        best_model.predict(test_dataloader, mode=("raw", "prediction"), n_samples=None)  # type: ignore
    )
    .base_dist.covariance_matrix.mean(0)  # type: ignore
    .cpu()
)

corr = cov / cov.diag().outer(cov.diag()).sqrt()

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

# Correlation matrix
fig = plt.figure()
plt.imshow(corr, cmap="bwr", vmin=-1, vmax=1)
plt.colorbar()
summary_writer.add_figure("correlation", fig)

# Correlations histogram
fig = plt.figure()
plt.hist(corr[corr < 1], edgecolor="black")
summary_writer.add_figure("correlation_histogram", fig)
