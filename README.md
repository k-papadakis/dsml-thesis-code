# Code for Forecasting Methods Thesis

For the Data Science and Machine Learning Master's of National Technical University of Athens.

Install and run `thesis -h` for more info.

To run all models for all datasets, use

```bash
thesis prophet electricity
thesis prophet traffic

thesis run electricity tft
thesis run traffic tft

thesis run electricity deepar
thesis run traffic deepar

thesis run electricity deepvar
thesis run traffic deepvar

thesis run electricity nbeats
thesis run traffic nbeats
```

To find hyperparams for all deep learning models, use

```bash
thesis find electricity tft --n-trials 30
thesis find traffic tft --n-trials 30

thesis find electricity deepar --n-trials 30
thesis find traffic deepar --n-trials 30

thesis find electricity nbeats --n-trials 30
thesis find traffic nbeats --n-trials 30
```

FIXME: Optuna doesn't work with DeepVar
