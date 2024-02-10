from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import DeepAR, MultivariateNormalDistributionLoss
from torch.utils.tensorboard.writer import SummaryWriter

from thesis.dataloading import SeriesDataModule
from thesis.metrics import METRICS

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")

ROOT_DIR = Path("output", "traffic", "deepvar")

datamodule = SeriesDataModule(
    name="traffic",
    path="./datasets/traffic/",
    multivariate=True,
    with_covariates=True,
    transform="auto",
    batch_size=50,  # 50 parallel time series
)
datamodule.setup("test")
test_dataset = datamodule.test

# model
model = DeepAR.from_dataset(
    test_dataset,
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
    datamodule=datamodule,
)
_ = trainer.test(ckpt_path="best", datamodule=datamodule)

# load best
best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore
best_model = DeepAR.load_from_checkpoint(best_model_path)

# predict
out = best_model.predict(
    test_dataset,
    mode="raw",
    return_x=True,
    return_y=True,
    return_index=True,
    n_samples=100,
)

# Correlation matrix of the average prediction random variable (84 predictions)
cov = (
    best_model.loss.map_x_to_distribution(
        best_model.predict(test_dataset, mode=("raw", "prediction"), n_samples=None)  # type: ignore
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
