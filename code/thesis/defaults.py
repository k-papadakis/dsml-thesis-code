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


def electricity_nbeats() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-4,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
    )
    model_config = NBEATSConfig(
        expansion_coefficient_lengths=[3, 2],
        widths=[256, 2048],
    )

    setting = nbeats("electricity", model_config, training_config)
    return setting


def traffic_nbeats() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-4,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
    )
    model_config = NBEATSConfig(
        expansion_coefficient_lengths=[3, 2],
        widths=[256, 2048],
    )

    setting = nbeats("traffic", model_config, training_config)
    return setting


def electricity_deepvar() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-2,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
    )
    model_config = DeepARConfig(
        hidden_size=30,
        rnn_layers=2,
        distribution="multinormal",
    )

    setting = deepar("electricity", model_config, training_config)
    return setting


def traffic_deepvar() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-2,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
    )
    model_config = DeepARConfig(
        hidden_size=30,
        rnn_layers=2,
        distribution="multinormal",
    )

    setting = deepar("traffic", model_config, training_config)
    return setting


def traffic_deepar() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-2,
        gradient_clip_val=0.1,
        dropout=0.1,
        max_epochs=100,
    )
    model_config = DeepARConfig(
        hidden_size=30,
        rnn_layers=2,
        distribution="beta",
    )
    setting = deepar("traffic", model_config, training_config)
    return setting


def electricity_tft() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-3,
        gradient_clip_val=0.01,
        dropout=0.1,
        max_epochs=100,
    )

    model_config = TFTConfig(
        hidden_size=160,
        lstm_layers=1,
        attention_head_size=4,
    )

    setting = tft("electricity", model_config, training_config)
    return setting


def traffic_tft() -> Setting:
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-3,
        gradient_clip_val=100.0,
        dropout=0.3,
        max_epochs=100,
    )

    model_config = TFTConfig(
        hidden_size=320,
        lstm_layers=1,
        attention_head_size=4,
    )

    setting = tft("traffic", model_config, training_config)
    return setting
