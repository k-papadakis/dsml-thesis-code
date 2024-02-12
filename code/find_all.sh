thesis find electricity tft --n-trials 30 --seed 42 --storage sqlite:///experiments2.db
thesis find traffic tft --n-trials 30 --seed 42 --storage sqlite:///experiments2.db

thesis find electricity deepar --n-trials 30 --seed 42 --storage sqlite:///experiments2.db
thesis find traffic deepar --n-trials 30 --seed 42 --storage sqlite:///experiments2.db

# FIXME: Optuna doesn't work with DeepVar

thesis find electricity nbeats --n-trials 30 --seed 42 --storage sqlite:///experiments2.db
thesis find traffic nbeats --n-trials 30 --seed 42 --storage sqlite:///experiments2.db
