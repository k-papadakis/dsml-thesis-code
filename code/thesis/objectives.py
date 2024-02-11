from typing import Literal

import lightning.pytorch as pl
import optuna
from lightning.pytorch.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from .configs import (
    DeepARConfig,
    NBEATSConfig,
    Setting,
    TFTConfig,
    TrainingConfig,
    deepar,
    nbeats,
    tft,
)


def _common_training_config(trial: optuna.Trial) -> TrainingConfig:
    return TrainingConfig(
        batch_size=128,
        max_epochs=10,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        gradient_clip_val=trial.suggest_float(
            "gradient_clip_val", 0.1, 100.0, log=True
        ),
        dropout=trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
    )


def _modify_setting(setting: Setting, trial: optuna.Trial) -> None:
    setting.trainer = pl.Trainer(
        max_epochs=setting.trainer.max_epochs,
        gradient_clip_val=setting.trainer.gradient_clip_val,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=2,
                mode="min",
                verbose=False,
            ),
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ],
        gradient_clip_algorithm="norm",
        limit_train_batches=0.2,
        limit_val_batches=0.25,
        # logging
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )
    setting.model.hparams.log_interval = -1  # type: ignore
    setting.model.hparams.log_val_interval = -1  # type: ignore


def nbeats_objective(dataset_name: Literal["electricity", "traffic"]):
    def inner(trial: optuna.Trial) -> float:

        model_config = NBEATSConfig(
            expansion_coefficient_lengths=[3, 2],
            widths=[256, trial.suggest_categorical("seasonal", [512, 2048])],
        )

        training_config = _common_training_config(trial)
        setting = nbeats(dataset_name, model_config, training_config)
        _modify_setting(setting, trial)

        setting.fit()

        return setting.trainer.callback_metrics["val_loss"].item()

    return inner


def tft_objective(dataset_name: Literal["electricity", "traffic"]):
    def inner(trial: optuna.Trial) -> float:

        model_config = TFTConfig(
            hidden_size=trial.suggest_categorical("hidden_size", [80, 160, 320]),
            lstm_layers=1,
            attention_head_size=4,
        )

        training_config = _common_training_config(trial)
        setting = tft(dataset_name, model_config, training_config)
        _modify_setting(setting, trial)

        setting.fit()

        return setting.trainer.callback_metrics["val_loss"].item()

    return inner


def deepar_objective(
    dataset_name: Literal["electricity", "traffic"],
    distribution: Literal["multinormal", "normal", "beta"],
):
    if dataset_name == "electricity" and distribution == "beta":
        raise ValueError(
            f"{dataset_name} dataset does not support Beta distribution,"
            "because its values are not in [0, 1]"
        )

    def inner(trial: optuna.Trial) -> float:

        model_config = DeepARConfig(
            hidden_size=trial.suggest_categorical("hidden_size", [10, 30]),
            rnn_layers=2,
            distribution=distribution,
        )

        training_config = _common_training_config(trial)
        setting = deepar(dataset_name, model_config, training_config)
        _modify_setting(setting, trial)

        setting.fit()

        return setting.trainer.callback_metrics["val_loss"].item()

    return inner
