# Smart-Home-Electricity-Demand-
Smart Home Electricity Demand 
# Predicting Single-Home Electricity Demand Using Support Vector Regression and Time-Series Forecasting

## Project Overview

This project focuses on predicting electricity demand for a single residential home using Support Vector Regression (SVR) and time-series forecasting techniques.

Electricity consumption varies throughout the day depending on household activity, environmental conditions, and historical usage patterns. Understanding these patterns can help homeowners anticipate electricity demand and make more informed decisions about energy efficiency.

The primary objective of this project is to develop a machine learning model that uses historical electricity consumption, time-related features, and available environmental measurements to predict a home's electricity demand **one hour into the future**.

The project will also include an interactive Streamlit dashboard that allows users to visualize historical electricity consumption and the model's predictions.

## Problem Statement

Residential electricity demand is not constant. Household electricity consumption changes depending on factors such as the time of day, temperature, weather conditions, and daily activities.

Accurately forecasting electricity demand can help homeowners better understand consumption patterns and anticipate periods of increased electricity usage.

This project investigates the following question:

**How accurately can Support Vector Regression predict a single home's electricity demand one hour ahead using historical electricity consumption and environmental data?**

## Project Objectives

The primary objectives of this project are to:

1. Analyze historical electricity consumption data from a single residential home.
2. Identify relevant time-related and environmental features that may influence electricity demand.
3. Apply time-series feature engineering to transform historical electricity measurements into supervised learning data.
4. Develop a Support Vector Regression model to predict electricity demand one hour ahead.
5. Evaluate the model's predictive performance using regression evaluation metrics.
6. Investigate how different features and SVR hyperparameters affect prediction accuracy.
7. Develop an interactive Streamlit dashboard to visualize electricity consumption and forecast results.

## Dataset

The project uses the **Smart Home Dataset with Weather Information**, publicly available on Kaggle.

**Dataset source:**

https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information

The dataset contains historical smart home electricity measurements along with associated weather information.

The prediction target will be overall household electricity demand, measured in kilowatts (kW).

The dataset will be examined to identify its exact columns, timestamp format, sampling frequency, missing values, and other preprocessing requirements.

### Data Preparation

The dataset will undergo the following preprocessing steps:

1. Load and inspect the dataset using Pandas.
2. Identify the timestamp and overall household electricity demand columns.
3. Check for missing values, duplicate timestamps, and inconsistencies.
4. Convert the dataset into hourly observations if necessary.
5. Create time-related and historical consumption features.
6. Separate the data chronologically into training, validation, and testing sets.
7. Apply appropriate feature scaling using parameters learned from the training data.

## Methodology

### 1. Time-Series Forecasting

Time-series forecasting uses historical observations to predict future values.

In this project, historical electricity consumption measurements will be used to predict household electricity demand one hour ahead.

The dataset will be transformed into a supervised learning problem through feature engineering.

Potential features include:

- Hour of the day
- Day of the week
- Electricity demand one hour earlier
- Electricity demand two hours earlier
- Electricity demand three hours earlier
- Electricity demand at the same hour on the previous day
- Rolling averages of historical electricity demand
- Available environmental measurements such as temperature and humidity

Only information available at the time of prediction will be used as model inputs to prevent data leakage.

### 2. Support Vector Regression (SVR)

Support Vector Regression is a supervised machine learning algorithm used to predict continuous numerical values.

Unlike ordinary linear regression, SVR can use kernel functions to model nonlinear relationships between input features and the target variable.

This project will initially use the Radial Basis Function (RBF) kernel to investigate nonlinear relationships between historical electricity demand, time-related features, and environmental conditions.

The model will be implemented using Scikit-learn.

The following hyperparameters will be investigated:

- **C:** Controls the trade-off between model complexity and prediction errors.
- **Epsilon:** Determines the width of the error-tolerance region around predictions.
- **Gamma:** Controls the influence of individual training observations when using the RBF kernel.

Feature scaling will be applied because SVR is sensitive to differences in feature magnitudes.

### 3. Model Training and Validation

The dataset will be divided chronologically into training, validation, and testing sets.

Unlike random splitting, chronological splitting preserves the order of observations and better represents real-world forecasting.

The training set will be used to fit the SVR model.

The validation set will be used to compare hyperparameter configurations and select the preferred model.

The held-out test set will be used to evaluate the selected model's performance on unseen future observations.

### 4. Model Evaluation

The model will be evaluated using the following regression metrics:

**Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted electricity demand.

**Root Mean Squared Error (RMSE):** Measures prediction error while placing greater emphasis on larger errors.

**R-squared (R²):** Measures predictive performance relative to a constant-mean baseline.

Model predictions will also be compared against actual electricity demand through visualizations.

Additional experiments will investigate whether including environmental measurements and different historical lag features improves forecasting performance.

## Interactive Dashboard

An interactive graphical interface will be developed using Streamlit.

The dashboard is planned to allow users to:

- View historical electricity consumption.
- Select an available historical forecasting point.
- Generate a one-hour-ahead electricity demand prediction.
- Compare predicted demand with actual recorded demand when available.
- Visualize prediction results through charts.

The dashboard will use the trained SVR model and the same preprocessing transformations used during model training.

## Technologies and Libraries

The project will use the following technologies:

- **Python:** Primary programming language
- **Pandas:** Data manipulation and preprocessing
- **NumPy:** Numerical computations
- **Scikit-learn:** SVR implementation, preprocessing, and evaluation
- **Matplotlib:** Data visualization
- **Streamlit:** Interactive dashboard
- **Joblib:** Saving and loading the trained model

## Repository Structure

The planned repository structure is:

```text
smart-home-energy-forecasting/
├── data/
├── src/
├── models/
├── documents/
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

The `data` directory will contain the dataset files.

The `src` directory will contain the Python scripts for data exploration, preprocessing, model training, and evaluation.

The `models` directory will contain the saved trained model and required preprocessing objects.

The `documents` directory will contain figures and diagrams used in the final report.

The `app.py` file will contain the Streamlit dashboard.

Additional scripts and files will be documented as they are implemented.

## Installation

Detailed installation instructions will be provided once the project dependencies and scripts have been implemented and tested.

The final installation instructions will include:

1. Cloning the GitHub repository.
2. Creating and activating a Python virtual environment.
3. Installing the required libraries using `requirements.txt`.
4. Downloading and placing the dataset in the appropriate directory.

Exact dependency versions and platform-specific instructions will be documented.

## Training and Evaluation

Commands for preprocessing the dataset, training the SVR model, performing validation, and evaluating the final model will be added after the corresponding scripts have been implemented.

The final documentation will explain how to reproduce the experiments and identify the location of the training, validation, and test data.

## Running the Streamlit Dashboard

Instructions for launching the interactive dashboard will be added once the application and trained model are available.

The final README will include the exact launch command and any required model files.

## Project Limitations

The model is designed to forecast electricity demand for a single home. Its predictions may not generalize to other households with different consumption patterns.

Prediction accuracy will depend on the quality and availability of historical electricity measurements, environmental information, and the selected time-series features.

The project will focus on one-hour-ahead forecasting rather than long-term electricity demand prediction.

## Project Status

**In Development**

Current phase: Repository setup and dataset exploration.

## References

Smart Home Dataset with Weather Information. Kaggle.

https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information

Combining Embeddings and Fuzzy Time Series for High-Dimensional Time Series Forecasting in Internet of Energy Applications.

https://www.researchgate.net/publication/356818145_Combining_Embeddings_and_Fuzzy_Time_Series_for_High-Dimensional_Time_Series_Forecasting_in_Internet_of_Energy_Applications
