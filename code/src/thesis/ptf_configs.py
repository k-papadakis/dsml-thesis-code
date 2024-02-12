from os import PathLike

from .ptf_api import (
    DeepARConfig,
    DeepVARConfig,
    NBEATSConfig,
    Setting,
    TFTConfig,
    TrainingConfig,
    deepar,
    deepvar,
    nbeats,
    tft,
)


def electricity_nbeats(
    input_dir: str | PathLike[str], output_dir: str | PathLike[str]
) -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=5e-4,
        gradient_clip_val=1.0,
        dropout=0.2,
    )
    model_config = NBEATSConfig(
        expansion_coefficient_lengths=[3, 2],
        widths=[64, 512],
    )

    setting = nbeats(
        "electricity", model_config, training_config, input_dir, output_dir
    )
    return setting


def traffic_nbeats(
    input_dir: str | PathLike[str], output_dir: str | PathLike[str]
) -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=7e-4,
        gradient_clip_val=30.0,
        dropout=0.1,
    )
    model_config = NBEATSConfig(
        expansion_coefficient_lengths=[3, 2],
        widths=[64, 512],
    )

    setting = nbeats("traffic", model_config, training_config, input_dir, output_dir)
    return setting


def electricity_deepvar(
    input_dir: str | PathLike[str], output_dir: str | PathLike[str]
) -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-3,
        gradient_clip_val=2.0,
        dropout=0.1,
    )
    model_config = DeepVARConfig(
        hidden_size=120,
        rnn_layers=2,
        rank=50,
    )

    setting = deepvar(
        "electricity", model_config, training_config, input_dir, output_dir
    )
    return setting


def electricity_deepar(
    input_dir: str | PathLike[str], output_dir: str | PathLike[str]
) -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-3,
        gradient_clip_val=2.0,
        dropout=0.1,
    )
    model_config = DeepARConfig(
        hidden_size=120,
        rnn_layers=2,
        distribution="normal",
    )

    setting = deepar(
        "electricity", model_config, training_config, input_dir, output_dir
    )
    return setting


def traffic_deepvar(
    input_dir: str | PathLike[str], output_dir: str | PathLike[str]
) -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-2,
        gradient_clip_val=40.0,
        dropout=0.1,
    )
    model_config = DeepVARConfig(
        hidden_size=120,
        rnn_layers=2,
        rank=50,
    )

    setting = deepvar("traffic", model_config, training_config, input_dir, output_dir)
    return setting


def traffic_deepar(
    input_dir: str | PathLike[str], output_dir: str | PathLike[str]
) -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-2,
        gradient_clip_val=40.0,
        dropout=0.2,
    )
    model_config = DeepARConfig(
        hidden_size=120,
        rnn_layers=2,
        distribution="beta",
    )
    setting = deepar("traffic", model_config, training_config, input_dir, output_dir)
    return setting


def electricity_tft(
    input_dir: str | PathLike[str], output_dir: str | PathLike[str]
) -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=3e-3,
        gradient_clip_val=50.0,
        dropout=0.2,
    )

    model_config = TFTConfig(
        hidden_size=160,
        lstm_layers=1,
        attention_head_size=4,
    )

    setting = tft("electricity", model_config, training_config, input_dir, output_dir)
    return setting


def traffic_tft(
    input_dir: str | PathLike[str], output_dir: str | PathLike[str]
) -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-2,
        gradient_clip_val=1.0,
        dropout=0.3,
    )

    model_config = TFTConfig(
        hidden_size=80,
        lstm_layers=1,
        attention_head_size=4,
    )

    setting = tft("traffic", model_config, training_config, input_dir, output_dir)
    return setting