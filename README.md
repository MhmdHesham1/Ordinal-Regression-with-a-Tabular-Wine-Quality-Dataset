Wine Quality Prediction Project

Overview

This project aims to predict the quality of wine based on various physicochemical properties. It uses machine learning models to analyze features such as acidity, alcohol content, and density to estimate wine quality on a scale.

Dataset

The dataset used in this project is from the Kaggle Playground Series, Season 3, Episode 5. It includes the following files:

train.csv: Training data with features and quality ratings
test.csv: Test data for predictions
sample_submission.csv: Example of the submission format

Features

The main features used for prediction include:

Fixed acidity
Volatile acidity
Residual sugar
Chlorides
Free sulfur dioxide
Total sulfur dioxide
Density
pH
Sulphates
Alcohol

Note: 'Citric acid' was excluded from the final model.
Models Explored
Several regression models were implemented and evaluated:

Random Forest Regressor
Gradient Boosting Regressor
Decision Tree Regressor
K-Nearest Neighbors Regressor
XGBoost Regressor
CatBoost Regressor
LightGBM Regressor
Voting Regressor (ensemble of the above models)

Evaluation Metrics

The models were evaluated using the following metrics:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R2 Score

Best Performing Model

Based on the evaluation metrics, the Gradient Boosting Regressor showed the best performance:

Mean Squared Error: 0.4935
Mean Absolute Error: 0.5250
R2 Score: 0.2805

However, the final submission uses the LightGBM model for predictions.

Usage

Ensure you have the required libraries installed (numpy, pandas, scikit-learn, xgboost, catboost, lightgbm).
Load the training data and preprocess it (scaling features).
Train the models and evaluate their performance.
Use the best performing model to make predictions on the test set.
Round the predictions to the nearest integer and save the results in the required submission format.

Future Improvements


Feature engineering to create more informative features
Hyperparameter tuning for each model
Exploring other ensemble methods or neural networks
Collecting more data or incorporating domain knowledge to improve predictions

Submission
The final predictions are rounded to the nearest integer and saved in a CSV file named 'submission.csv' in the format required by the competition.
