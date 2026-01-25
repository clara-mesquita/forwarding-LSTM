#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# It is recommended to run this script with the virtual environment activated.
# For example: source .venv/bin/activate

echo "Running GRU experiment..."
python code/forwarding-LSTM/grid/gru.py

echo "Running LSTM experiment..."
python code/forwarding-LSTM/grid/lstm.py

echo "Running SARIMA experiment..."
python code/forwarding-LSTM/grid/sarima.py

echo "Running Recursive LSTM experiment..."
python code/forwarding-LSTM/grid/recursive_lstm_grid.py

echo "All experiments finished successfully."
