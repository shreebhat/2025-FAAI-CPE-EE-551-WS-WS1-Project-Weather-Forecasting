# 2025-FAAI-CPE-EE-551-WS-WS1-Project-Weather-Forecasting

# Weather Forecasting Project – Data & Feature Pipeline

## Overview
This project implements the data ingestion and feature engineering pipeline for a daily weather
forecasting system. Public NOAA climate data is used to construct a clean time-series dataset and
generate features suitable for next-day temperature prediction.

The project is structured to separate data handling, feature generation, modeling, and evaluation
into modular components.



## Problem Description
Raw climate datasets often contain missing values, inconsistent formatting, and non-numeric
placeholders that must be handled before modeling. Accurate short-term weather forecasting depends
on reliable historical data and meaningful temporal features.

The goal of this portion of the project is to:
- load public daily climate data
- validate and clean the dataset
- generate time-series features
- prepare model-ready inputs in a reproducible way



## Dataset
We use publicly available NOAA daily climate data for **Albany, NY**, covering **January 2015 to
May 2022**.

Albany is used as a **representative Northeast U.S. climate station**, allowing the project to
capture seasonal weather patterns without introducing unnecessary spatial complexity.

**Source:** NOAA / National Centers for Environmental Information (NCEI)  
**File:** `data/daily_data.csv`

**Target variable:**
- `DailyAverageDryBulbTemperature`



## Project Structure
weather/
data.py # data loading, validation, and cleaning
features.py # feature engineering for time-series modeling

tests/
test_data.py # unit tests for data handling
test_features.py # unit tests for feature engineering

notebooks/
main.ipynb # main program entry point

data/
daily_data.csv




## Data Pipeline Design

### Data Loading and Validation
A `WeatherStation` class encapsulates:
- CSV file I/O
- required column validation
- date parsing and normalization
- numeric coercion of NOAA values
- daily deduplication

This ensures all downstream code operates on a clean, consistent dataset.

### Feature Engineering
Time-series features are generated using:
- lagged temperature values
- rolling means and standard deviations
- calendar-based features (month, weekday)
- cyclic encoding of yearly seasonality (sin/cos)

Rolling statistics are shifted to prevent data leakage.



## Testing
Pytest is used to validate:
- correct handling of missing or malformed input
- feature creation correctness
- generator behavior and ordering
- expected data shapes

All tests pass using:

python -m pytest

## How to Run

Install dependencies:

pip install pandas numpy pytest


Run unit tests:

python -m pytest



Open and run the main notebook:

notebooks/main.ipynb



The notebook produces feature matrix X and target vector y for downstream modeling.