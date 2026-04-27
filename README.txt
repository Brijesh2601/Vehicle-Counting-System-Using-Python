
# Vehicle Counter Project - Setup Guide

## 1. Install Dependencies
Open your terminal (Command Prompt) in this folder and run:

pip install -r requirements.txt
pip install ultralytics opencv-python-headless matplotlib plotly numpy pandas

## 2. Run the App
Run the provided PowerShell script to start the dashboard safely (avoids library conflicts):

./run_app.ps1

Alternatively, if you are using CMD:
set PYTHONPATH=
python -m streamlit run app.py

## Note about 'libs' folder
This folder contains libraries pre-compiled for a different Python version. 
We have updated the app to use system-installed libraries instead for better compatibility.
