#!/bin/bash

python -m thesis.prophet_electricity
python -m thesis.prophet_traffic

python -m thesis.runner electricity nbeats
python -m thesis.runner electricity deepvar
python -m thesis.runner electricity tft
python -m thesis.runner traffic nbeats
python -m thesis.runner traffic deepvar
python -m thesis.runner traffic deepar
python -m thesis.runner traffic tft