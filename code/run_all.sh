#!/bin/bash

python prophet_electricity.py
python prophet_traffic.py

python runner.py electricity nbeats
python runner.py electricity deepvar
python runner.py electricity tft
python runner.py traffic nbeats
python runner.py traffic deepvar
python runner.py traffic deepar
python runner.py traffic tft