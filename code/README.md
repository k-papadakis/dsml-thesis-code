# Thesis Code

Install and run `thesis -h` for more info.

To run all models for all datasets, use

```bash
thesis prophet electricity
thesis prophet traffic

thesis run electricity tft --seed 42
thesis run traffic tft --seed 42

thesis run electricity deepar --seed 42
thesis run traffic deepar --seed 42

thesis run electricity deepvar --seed 42
thesis run traffic deepvar --seed 42

thesis run electricity nbeats --seed 42
thesis run traffic nbeats --seed 42
```

To find hyperparams for all deep learning models, use

```bash
thesis find electricity tft --n-trials 30 --seed 42
thesis find traffic tft --n-trials 30 --seed 42

thesis find electricity deepar --n-trials 30 --seed 42
thesis find traffic deepar --n-trials 30 --seed 42

thesis find electricity nbeats --n-trials 30 --seed 42
thesis find traffic nbeats --n-trials 30 --seed 42
```

FIXME: Optuna doesn't work with DeepVar
