python -m thesis.prophet_electricity
python -m thesis.prophet_traffic

python -m thesis.runner electricity nbeats --seed 42
python -m thesis.runner electricity deepvar --seed 42
python -m thesis.runner electricity tft --seed 42
python -m thesis.runner traffic nbeats --seed 42
python -m thesis.runner traffic deepvar --seed 42
python -m thesis.runner traffic deepar --seed 42
python -m thesis.runner traffic tft --seed 42
