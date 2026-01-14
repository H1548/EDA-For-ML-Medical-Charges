# Exploratory Data Analysis For Machine Learning - Medical Charges Dataset

Conducting exploratory data analysis to better understand the dataset and the features involved, then training ML models to become medical charges predictors and analysing the best performing model. 

## Project Overview

The goal of this project is to explore the dataset provided, to understand specific features, their distributions and possible relationship with the target, which is the final medical charge. 
This information will be used to model the dataset so that a machine learning model can effectively train and learn underlying patterns and relationships within the data, with stability. The task that the model is training on is to predict the medical charge that a patient will incur given their medical record which consists of: 'age', 'sex, 'bmi', 'children', 'smoker', 'region'. Becuase the targets are seen as continuos values, this will be a regression task, which means 3 regression models will be trained, they are: the linear regression model, the random forest regressor and the XGboost regressor. Once the model's have been trained there performance will be compared via two metrics which are the Mean Square Error (MSE) and the Mean Absolute Error (MAE). 

## Dataset 
- Source: Kaggle – Medical Insurance Cost Prediction
- Rows: 1,338
- Features:
    - Numerical: age, bmi, children, charges
    - Categorical: sex, smoker, region
- Target Variable: charges

## Analysis Performed
- Type Inspection
- Univariate analysis (Value counts, distributions, skewness)
- Bivariate analysis (Indvidual features vs target)
- Target Transformation (Log-Transformation)
- Visual eploration using boxplots, scatter plots, stripplot, and countplots

## Key insights
- Target Distribution has right skew of value 1.5, transformation such as the log transormation can help the distribution become more uniform, providing more stability in ML training
- Smoking has a strong impact on charges
- Age is positively correlated with charges but not linearly across all groups, however, this features does share a non-linear relationship with charges

## Results
### Baseline Predictor (Mean of all targets)
MSE: 155391443.6, MAE: 9593
### Linear Regression
- Original target distribution: MSE: 33596915.8, MAE: 4181
- Log-transformed target distribution: MSE: 61079027.7, MAE: 3888.7
### Random Forest Algorithm
- Original target distribution: MSE: 21938457, MAE: 2551
- Log-transformed target distribution: MSE: 19381110.7, MAE: 2063
### XGboost 
- Original target distribution: MSE: 18726613, MAE: 2402
- Log-transformed target distribution: MSE: 20610669.7, MAE: 2107

## Notebook Contents
- 'MedicalChargesPredictor.ipynb': Complete notebook with eda, explanations, visualizations, ML training and analyses and conclusions

## How to Run 
```bash
pip install -r requirements.txt
jupyter notebook MedicalChargePredictor.ipynb
```

## Next Steps 
- Potentially scale the dataset to include more training samples
- Scale and complexify ML model, maybe use MLP to understand complex realtionships
