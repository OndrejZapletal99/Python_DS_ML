# Python_DS_ML

## Table of Contents
- [Python\_DS\_ML](#python_ds_ml)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Draw Data - Interactive Classification Playground](#draw-data---interactive-classification-playground)
  - [Demand Forecast](#demand-forecast)
  - [Flight Price Prediction](#flight-price-prediction)
  - [Health Insurance Analysis](#health-insurance-analysis)
  - [Heart Disease Prediction](#heart-disease-prediction)
  - [Calculator App](#calculator-app)
  - [Solar – PV Output Prediction](#solar--pv-output-prediction)

## Introduction

This repository is dedicated to various areas of data science and machine learning. Here you will find code examples, projects, tutorials, and notes covering topics such as data processing, visualization, modeling, machine learning, neural networks, and more. The goal is to provide a practical resource for both learning and quick reference in Python for data science and machine learning.

## Draw Data - Interactive Classification Playground

The `Draw Data` notebook demonstrates how to use the [drawdata](https://github.com/koaning/drawdata) library to interactively create and classify your own 2D datasets. With this tool, you can manually draw data points, assign them to classes, and instantly see how different machine learning classifiers (such as Logistic Regression, Random Forest, SVM, and KNN) separate your data.

**Key Features:**
- Interactive widget for drawing and labeling data points.
- Automatic encoding of class labels.
- Visualization of decision boundaries for multiple classifiers.
- Easy comparison of classifier performance on custom datasets.

**How it works:**
1. Use the interactive widget to draw points and assign them to different classes.
2. The notebook encodes your labels and prepares the data for modeling.
3. Several classifiers are trained on your drawn data.
4. For each classifier, the decision boundary is visualized along with your data points, so you can see how well each model separates the classes.

This playground is ideal for learning and teaching the basics of classification, understanding decision boundaries, and experimenting with how different algorithms behave on various data distributions.

## Demand Forecast

The `Demand Forecast` project focuses on predicting demand using historical data.

**Key Features:**
- Time series analysis for demand forecasting.
- Implementation of machine learning models for prediction.
- Data preprocessing and feature engineering for time series data.
- Evaluation of model performance using metrics like RMSE.

## Flight Price Prediction

The `Flight Price Prediction` project predicts flight ticket prices based on various parameters such as airline, class, and number of stops.

**Key Features:**
- Data cleaning and preprocessing for handling missing values and outliers.
- Feature engineering to extract meaningful insights from raw data.
- Implementation of regression models for price prediction.
- Comparison of model performance using metrics like MAE and R2.

## Health Insurance Analysis

The `Health Insurance Analysis` project analyzes and predicts health insurance costs based on demographic and health-related factors.

**Key Features:**
- Exploratory data analysis (EDA) to understand key factors affecting insurance costs.
- Implementation of regression models for cost prediction.
- Visualization of relationships between variables and insurance costs.
- Evaluation of model accuracy using metrics like MSE and R2.

## Heart Disease Prediction

The `Heart Disease Prediction` project predicts the presence of heart disease based on patient health data.

**Key Features:**
- Data preprocessing to handle missing values and categorical variables.
- Implementation of classification models for disease prediction.
- Visualization of feature importance and decision boundaries.
- Evaluation of model performance using metrics like accuracy, precision, and recall.

## Calculator App
The `Calculator App` predicts production costs for door and frame products based on user-defined specifications using a trained gradient boosting model.

**Key Features:**
- Interactive web app built with Streamlit for intuitive user input.
- Encodes categorical selections (e.g., type, category, color) into model-ready format.
- Uses a pre-trained LightGBM model to predict costs across various batch sizes.
- Displays results in a structured, dynamic table.
- Designed for scalability and integration with product configuration systems.

## Solar – PV Output Prediction

The `Solar` project focuses on predicting photovoltaic (PV) power plant output based on current weather forecasts for the Valašské Meziříčí location. It includes an interactive Streamlit web app that allows users to view production predictions from tomorrow up to the end of the available forecast and compare them with actual values from the previous year.

**Key Features:**
- Automatic loading and processing of current weather forecasts (temperature, cloud cover, precipitation, wind, etc.).
- Utilizes a pre-trained RandomForest model to predict daily PV output.
- Comparison of predicted values with real data for the same period last year.
- Visualization of results in clear charts and tables.
- Easily extendable to other locations or types of power plants.

**How it works:**
1. The app loads the latest weather forecast from an API (e.g., Visual Crossing).
2. Data is preprocessed and encoded using a saved encoder.
3. The model predicts PV output for each day in the forecast.
4. Results are displayed in the app and compared with historical data.