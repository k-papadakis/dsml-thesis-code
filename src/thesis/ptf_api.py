from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable, Literal, Optional

import lightning.pytorch as pl
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner.lr_finder import _LRFinder
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.utilities.parsing import lightning_setattr
from matplotlib import pyplot as plt
from pytorch_forecasting import MAE, MAPE, MASE, RMSE, SMAPE
from pytorch_forecasting import BaseModel as ForecastingModel
from pytorch_forecasting import (
    BaseModelWithCovariates as ForecastingModelWithCovariates,
)
from pytorch_forecasting import (
    BetaDistributionLoss,
    DeepAR,
    EncoderNormalizer,
    MultivariateNormalDistributionLoss,
    NBeats,
    NormalDistributionLoss,
    QuantileLoss,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from .dataloading import (
    ELECTRICITY_URL,
    TRAFFIC_URL,
    download_and_extract_zip,
    load_electricity,
    load_traffic,
)
from .metrics import compute_metrics


class SeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: Literal["electricity", "traffic"],
        path: str | PathLike,
        multivariate: bool,
        with_covariates: bool,
        transform: Literal["auto", "logit"],
        batch_size: int,
    ):
        super().__init__()

        self.name = name
        self.path = Path(path)
        self.multivariate = multivariate
        self.with_covariates = with_covariates
        self.transform = transform
        self.batch_size = batch_size

    def prepare_data(self):
        if self.path.is_dir():
            return

        if self.name == "electricity":
            url = ELECTRICITY_URL
        elif self.name == "traffic":
            url = TRAFFIC_URL
        else:
            raise ValueError(f"Unknown dataset name: {self.name}")

        print(f"Dataset {self.name} not found in {self.path}")
        print(f"Downloading dataset {self.name} from {url} to {self.path}")
        download_and_extract_zip(url, self.path)
        print(f"Downloaded dataset {self.name} to {self.path}")

    def setup(self, stage: str):
        if self.name == "electricity":
            loader = load_electricity
        elif self.name == "traffic":
            loader = load_traffic
        else:
            raise ValueError(f"Unknown dataset name: {self.name}")

        data = loader(self.path)
        freq = pd.Timedelta(data.index.freq)  # type: ignore

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

        horizon = pd.Timedelta(1, "day")

        output_length = horizon // freq
        input_length = 7 * output_length
        validation_cutoff = data["time_idx"].max() - output_length
        training_cutoff = validation_cutoff - 21 * output_length

        if self.transform == "logit":
            target_normalizer = EncoderNormalizer(
                method="identity", transformation="logit"
            )
        elif self.transform == "auto":
            target_normalizer = "auto"
        else:
            raise ValueError(f"Unknown transform method: {self.transform}")

        self.train = TimeSeriesDataSet(
            data[data["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target="value",
            group_ids=["series"],
            time_varying_unknown_reals=["value"],
            max_encoder_length=input_length,
            max_prediction_length=output_length,
            time_varying_known_categoricals=(
                ["hour", "weekday"] if self.with_covariates else []
            ),
            static_categoricals=["series"] if self.with_covariates else [],
            target_normalizer=target_normalizer,
        )
        self.val = TimeSeriesDataSet.from_dataset(
            self.train,
            data[data["time_idx"] <= validation_cutoff],
            min_prediction_idx=training_cutoff + 1,
        )
        self.test = TimeSeriesDataSet.from_dataset(
            self.train,
            data,
            # min_prediction_idx=validation_cutoff + 1,
            predict=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.train.to_dataloader(
            train=True,
            batch_sampler="synchronized" if self.multivariate else None,  # type: ignore
            batch_size=self.batch_size,
            num_workers=2,
        )

    def val_dataloader(self) -> DataLoader:
        return self.val.to_dataloader(
            train=False,
            batch_sampler="synchronized" if self.multivariate else None,  # type: ignore
            batch_size=self.batch_size,
            num_workers=2,
        )

    def test_dataloader(self) -> DataLoader:
        return self.test.to_dataloader(
            train=False,
            batch_sampler="synchronized" if self.multivariate else None,  # type: ignore
            batch_size=self.batch_size,
            num_workers=0,
        )


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    gradient_clip_val: float
    dropout: float


class ModelConfig:
    pass


@dataclass
class NBEATSConfig(ModelConfig):
    expansion_coefficient_lengths: list[int]
    widths: list[int]


@dataclass
class TFTConfig(ModelConfig):
    hidden_size: int
    lstm_layers: int
    attention_head_size: int


@dataclass
class DeepVARConfig(ModelConfig):
    hidden_size: int
    rnn_layers: int
    rank: int


@dataclass
class DeepARConfig(ModelConfig):
    hidden_size: int
    rnn_layers: int
    distribution: Literal["beta", "normal"]


@dataclass
class Setting:
    datamodule: SeriesDataModule
    model: ForecastingModel
    trainer: pl.Trainer

    def find_lr(self, min_lr=1e-8, max_lr=1.0) -> Optional[float]:
        lr_finder: Optional[_LRFinder] = Tuner(self.trainer).lr_find(
            self.model,
            datamodule=self.datamodule,
            min_lr=min_lr,
            max_lr=max_lr,
            update_attr=False,
        )
        assert lr_finder is not None
        return lr_finder.suggestion()

    def set_lr(self, lr: float) -> None:
        lightning_setattr(self.model, "learning_rate", lr)

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
            cov: torch.Tensor = (
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
            corr = pd.DataFrame(
                corr.numpy(),
                index=out.index["series"],
                columns=out.index["series"],
            )

            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(
                corr,
                center=0.0,
                annot=False,
                cmap="coolwarm",
                linewidths=0.5,
                cbar_kws={"label": "Correlation Coefficient"},
            )
            plt.tight_layout()
            summary_writer.add_figure("correlation_heatmap", fig)

            fig = plt.figure()
            sns.histplot(corr.values[corr.values < 1.0], kde=True)
            plt.xlabel("Correlation Coefficient")
            plt.tight_layout()
            summary_writer.add_figure("correlation_histogram", fig)

        # TFT, DeepAR, DeepVAR
        elif isinstance(best_model, ForecastingModelWithCovariates):
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
    input_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
) -> Setting:

    input_dir = Path(input_dir, name)
    output_dir = Path(output_dir, name, "nbeats")

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
        patience=5,
        mode="min",
        verbose=False,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False)
    trainer = pl.Trainer(
        max_epochs=100,
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
    input_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
) -> Setting:

    input_dir = Path(input_dir, name)
    output_dir = Path(output_dir, name, "deepar")

    datamodule = SeriesDataModule(
        name=name,
        path=input_dir,
        multivariate=False,
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
        logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]),
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        mode="min",
        verbose=False,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False)
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        default_root_dir=str(output_dir),
    )

    return Setting(datamodule, model, trainer)


def deepvar(
    name: Literal["electricity", "traffic"],
    model_config: DeepVARConfig,
    training_config: TrainingConfig,
    input_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
) -> Setting:

    input_dir = Path(input_dir, name)
    output_dir = Path(output_dir, name, "deepvar")

    datamodule = SeriesDataModule(
        name=name,
        path=input_dir,
        multivariate=True,
        with_covariates=True,
        transform="auto",
        batch_size=training_config.batch_size,
    )
    datamodule.prepare_data()
    datamodule.setup("test")
    test_dataset = datamodule.test

    quantiles = [0.1, 0.5, 0.9]
    loss = MultivariateNormalDistributionLoss(
        quantiles=quantiles, rank=model_config.rank
    )

    model: ForecastingModel = DeepAR.from_dataset(  # type: ignore
        test_dataset,
        learning_rate=training_config.learning_rate,
        hidden_size=model_config.hidden_size,
        rnn_layers=model_config.rnn_layers,
        loss=loss,
        dropout=training_config.dropout,
        log_interval=200,
        logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]),
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        mode="min",
        verbose=False,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False)
    trainer = pl.Trainer(
        max_epochs=100,
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
    input_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
) -> Setting:

    input_dir = Path(input_dir, name)
    output_dir = Path(output_dir, name, "tft")

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
        patience=5,
        mode="min",
        verbose=False,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=False)
    trainer = pl.Trainer(
        max_epochs=100,
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
        name: compute_metrics(
            y_true.cpu(),
            # the mean is for either the DeepAR samples or the TFT quantiles
            (y_pred.mean(-1) if y_pred.dim() == 2 else y_pred).cpu(),
        )
        for name, y_true, y_pred in zip(names, ys_true, ys_pred)
    }
    return pd.DataFrame(perf).T
