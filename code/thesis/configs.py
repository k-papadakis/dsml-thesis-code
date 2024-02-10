from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable, Literal, Optional

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from pytorch_forecasting import MAE, MAPE, MASE, RMSE, SMAPE
from pytorch_forecasting import BaseModel as ForecastingModel
from pytorch_forecasting import (
    BetaDistributionLoss,
    DeepAR,
    MultivariateNormalDistributionLoss,
    NBeats,
    NormalDistributionLoss,
    QuantileLoss,
    TemporalFusionTransformer,
)
from torch.utils.tensorboard.writer import SummaryWriter

from thesis.dataloading import SeriesDataModule
from thesis.metrics import METRICS


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    gradient_clip_val: float
    dropout: float
    max_epochs: int
    patience: int


@dataclass
class NBEATSConfig:
    expansion_coefficient_lengths: list[int]
    widths: list[int]


@dataclass
class TFTConfig:
    hidden_size: int
    lstm_layers: int
    attention_head_size: int


@dataclass
class DeepARConfig:
    hidden_size: int
    rnn_layers: int
    distribution: Literal["beta", "normal", "multinormal"]


@dataclass
class Setting:
    datamodule: SeriesDataModule
    model: ForecastingModel
    trainer: pl.Trainer

    def fit(self):
        self.trainer.fit(self.model, datamodule=self.datamodule)

    def test(self):
        self.trainer.test(ckpt_path="best", datamodule=self.datamodule)

    def summary_writer(self) -> SummaryWriter:
        return getattr(self.trainer.logger, "experiment")

    def log_dir(self) -> Path:
        assert self.trainer.log_dir is not None
        return Path(self.trainer.log_dir)

    def load_best(self) -> ForecastingModel:
        best_model_path = getattr(self.trainer.checkpoint_callback, "best_model_path")
        return type(self.model).load_from_checkpoint(best_model_path)

    def evaluate(self) -> None:
        # TODO: best_model.plot_prediction_actual_by_variable (BasemodelWithCovariates method)
        best_model = self.load_best()
        test_dataset = self.datamodule.test
        summary_writer = self.summary_writer()

        out = best_model.predict(
            test_dataset,
            mode="raw",
            return_x=True,
            return_y=True,
            return_index=True,
        )

        perf = performance(out.index["series"], out.y[0], out.output.prediction)
        perf.to_csv(self.log_dir() / "performance.csv")

        for i, name in out.index["series"].items():
            fig = best_model.plot_prediction(
                out.x,
                out.output,
                idx=i,
            )
            summary_writer.add_figure(f"prediction/{name}", fig)

        # cases for different models
        # NBEATS
        if isinstance(best_model, NBeats):
            for i, name in out.index["series"].items():
                fig = best_model.plot_interpretation(out.x, out.output, idx=i)
                summary_writer.add_figure(f"interpretation/{name}", fig)

        # DeepVAR
        elif isinstance(best_model, DeepAR) and isinstance(
            best_model.loss, MultivariateNormalDistributionLoss
        ):
            cov = (
                best_model.loss.map_x_to_distribution(
                    best_model.predict(
                        test_dataset,
                        mode=("raw", "prediction"),
                        n_samples=None,  # type: ignore
                    )
                )
                .base_dist.covariance_matrix.mean(0)  # type: ignore
                .cpu()
            )
            corr = cov / cov.diag().outer(cov.diag()).sqrt()

            fig = plt.figure()
            plt.imshow(corr, cmap="bwr", vmin=-1, vmax=1)
            plt.colorbar()
            summary_writer.add_figure("correlation", fig)

            fig = plt.figure()
            plt.hist(corr[corr < 1], edgecolor="black")
            summary_writer.add_figure("correlation_histogram", fig)

        # TFT
        elif isinstance(best_model, TemporalFusionTransformer):
            predictions_vs_actuals = best_model.calculate_prediction_actual_by_variable(
                out.x, out.output.prediction
            )
            figs = best_model.plot_prediction_actual_by_variable(predictions_vs_actuals)

            for name, fig in figs.items():  # type: ignore
                summary_writer.add_figure(f"prediction_actual_by_variable/{name}", fig)

    def run(self):
        self.fit()
        self.test()
        self.evaluate()


def nbeats(
    name: Literal["electricity", "traffic"],
    model_config: NBEATSConfig,
    training_config: TrainingConfig,
    input_dir: Optional[str | PathLike[str]] = None,
    output_dir: Optional[str | PathLike[str]] = None,
) -> Setting:

    if input_dir is None:
        input_dir = Path("datasets", name)

    if output_dir is None:
        output_dir = Path("output", name, "nbeats")

    datamodule = SeriesDataModule(
        name=name,
        path=input_dir,
        multivariate=False,
        with_covariates=False,
        transform="auto",
        batch_size=training_config.batch_size,
    )
    datamodule.prepare_data()
    datamodule.setup("test")
    test_dataset = datamodule.test

    model: ForecastingModel = NBeats.from_dataset(  # type: ignore
        test_dataset,
        expansion_coefficient_lengths=model_config.expansion_coefficient_lengths,
        widths=model_config.widths,
        learning_rate=training_config.learning_rate,
        dropout=training_config.dropout,
        log_interval=200,
        logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]),
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=training_config.patience,
        mode="min",
        verbose=False,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False)
    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        default_root_dir=str(output_dir),
    )

    return Setting(datamodule, model, trainer)


def deepar(
    name: Literal["electricity", "traffic"],
    model_config: DeepARConfig,
    training_config: TrainingConfig,
    input_dir: Optional[str | PathLike[str]] = None,
    output_dir: Optional[str | PathLike[str]] = None,
) -> Setting:
    if input_dir is None:
        input_dir = Path("datasets", name)

    if output_dir is None:
        output_dir = Path(
            "output",
            name,
            "deepvar" if model_config.distribution == "multinormal" else "deepar",
        )

    datamodule = SeriesDataModule(
        name=name,
        path=input_dir,
        multivariate=model_config.distribution == "multinormal",
        with_covariates=True,
        transform="auto" if model_config.distribution != "beta" else "logit",
        batch_size=training_config.batch_size,
    )
    datamodule.prepare_data()
    datamodule.setup("test")
    test_dataset = datamodule.test

    quantiles = [0.1, 0.5, 0.9]
    if model_config.distribution == "beta":
        loss = BetaDistributionLoss(quantiles=quantiles)
    elif model_config.distribution == "normal":
        loss = NormalDistributionLoss(quantiles=quantiles)
    elif model_config.distribution == "multinormal":
        loss = MultivariateNormalDistributionLoss(
            quantiles=quantiles, rank=model_config.hidden_size
        )
    else:
        raise ValueError(f"Unknown distribution: {model_config.distribution}")

    model: ForecastingModel = DeepAR.from_dataset(  # type: ignore
        test_dataset,
        learning_rate=training_config.learning_rate,
        hidden_size=model_config.hidden_size,
        rnn_layers=model_config.rnn_layers,
        loss=loss,
        dropout=training_config.dropout,
        log_interval=200,
        # TODO: Add quantiles loss here? Will it error?
        logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]),
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=training_config.patience,
        mode="min",
        verbose=False,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False)
    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        default_root_dir=str(output_dir),
    )

    return Setting(datamodule, model, trainer)


def tft(
    name: Literal["electricity", "traffic"],
    model_config: TFTConfig,
    training_config: TrainingConfig,
    input_dir: Optional[str | PathLike[str]] = None,
    output_dir: Optional[str | PathLike[str]] = None,
) -> Setting:
    if input_dir is None:
        input_dir = Path("datasets", name)

    if output_dir is None:
        output_dir = Path("output", name, "tft")

    datamodule = SeriesDataModule(
        name=name,
        path=input_dir,
        multivariate=False,
        with_covariates=True,
        transform="auto",
        batch_size=training_config.batch_size,
    )
    datamodule.prepare_data()
    datamodule.setup("test")
    test_dataset = datamodule.test

    quantiles = [0.1, 0.5, 0.9]
    model: ForecastingModel = TemporalFusionTransformer.from_dataset(  # type: ignore
        test_dataset,
        hidden_size=model_config.hidden_size,
        hidden_continuous_size=model_config.hidden_size,
        dropout=training_config.dropout,
        lstm_layers=model_config.lstm_layers,
        attention_head_size=model_config.attention_head_size,
        output_size=len(quantiles),
        learning_rate=training_config.learning_rate,
        log_interval=200,
        loss=QuantileLoss(quantiles),
        logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]),
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=training_config.patience,
        mode="min",
        verbose=False,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False)
    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        default_root_dir=str(output_dir),
    )

    return Setting(datamodule, model, trainer)


def performance(
    names: Iterable[str],
    ys_true: Iterable[torch.Tensor],
    ys_pred: Iterable[torch.Tensor],
) -> pd.DataFrame:
    perf = {
        metric_fn.__name__: {
            name: metric_fn(
                y_true.cpu(),
                # the mean is for either the DeepAR samples or the TFT quantiles
                (y_pred.mean(-1) if y_pred.dim() == 2 else y_pred).cpu(),
            ).item()
            for name, y_true, y_pred in zip(names, ys_true, ys_pred)
        }
        for metric_fn in METRICS
    }
    return pd.DataFrame(perf)


def default_electricity_nbeats() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-4,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
        patience=2,
    )
    model_config = NBEATSConfig(
        expansion_coefficient_lengths=[3, 2],
        widths=[256, 2048],
    )

    setting = nbeats("electricity", model_config, training_config)
    return setting


def default_traffic_nbeats() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-4,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
        patience=2,
    )
    model_config = NBEATSConfig(
        expansion_coefficient_lengths=[3, 2],
        widths=[256, 2048],
    )

    setting = nbeats("traffic", model_config, training_config)
    return setting


def default_electricity_deepvar() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-2,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
        patience=2,
    )
    model_config = DeepARConfig(
        hidden_size=30,
        rnn_layers=2,
        distribution="multinormal",
    )

    setting = deepar("electricity", model_config, training_config)
    return setting


def default_traffic_deepvar() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-2,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
        patience=2,
    )
    model_config = DeepARConfig(
        hidden_size=30,
        rnn_layers=2,
        distribution="multinormal",
    )

    setting = deepar("traffic", model_config, training_config)
    return setting


def default_traffic_deepar() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-2,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
        patience=2,
    )
    model_config = DeepARConfig(
        hidden_size=30,
        rnn_layers=2,
        distribution="beta",
    )
    setting = deepar("traffic", model_config, training_config)
    return setting


def default_electricity_tft() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-3,
        gradient_clip_val=0.01,
        dropout=0.1,
        max_epochs=100,
        patience=2,
    )

    model_config = TFTConfig(
        hidden_size=160,
        lstm_layers=1,
        attention_head_size=4,
    )

    setting = tft("electricity", model_config, training_config)
    return setting


def default_traffic_tft() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-3,
        gradient_clip_val=100.0,
        dropout=0.3,
        max_epochs=100,
        patience=2,
    )

    model_config = TFTConfig(
        hidden_size=320,
        lstm_layers=1,
        attention_head_size=4,
    )

    setting = tft("traffic", model_config, training_config)
    return setting
