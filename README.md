# Stock Data Analysis with Machine Learning Models

## Introduction
This project involves fetching and analyzing stock data using the `yfinance` library and applying various machine learning models for predicting stock prices.
The models include Support Vector Regression (SVR), XGBoost Regressor, and Gradient Boosting Regressor.

## Requirements
- Python 3.x
- `yfinance` library (install using `pip install yfinance`)
- `pandas` library
- `matplotlib` library
- `statsmodels` library for time series analysis (install using `pip install statsmodels`)
- `scikit-learn` library for data preprocessing, PCA, and various regression models (install using `pip install scikit-learn`)
- `seaborn` library for visualization (install using `pip install seaborn`)
- `xgboost` library for gradient boosting (install using `pip install xgboost`)

## Getting Started
1. Install the required libraries mentioned above.
2. Ensure you have a stable internet connection to fetch the stock data.
3. Clone this repository to your local machine.

## Usage
1. Import the required libraries:

```python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb

