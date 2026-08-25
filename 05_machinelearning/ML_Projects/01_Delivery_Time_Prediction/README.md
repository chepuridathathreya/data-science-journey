# Delivery Time Prediction System

A Machine Learning project that predicts the estimated food delivery time based on factors such as distance, weather, traffic conditions, preparation time, courier experience, time of day, and vehicle type.

## Problem Statement

Food delivery time can be affected by multiple real-world factors. This project aims to build a Machine Learning model that estimates delivery time based on available delivery-related information.

The goal is to demonstrate an end-to-end Machine Learning workflow, from data preprocessing and exploratory analysis to model comparison, evaluation, and prediction on new unseen data.

---

## Features Used

The model uses the following features:

- Distance (`Distance_km`)
- Weather
- Traffic Level
- Time of Day
- Vehicle Type
- Preparation Time (`Preparation_Time_min`)
- Courier Experience (`Courier_Experience_yrs`)

The target variable is:

- Delivery Time (`Delivery_Time_min`)

`Order_ID` was excluded because it does not provide meaningful predictive information.

---

## Project Workflow

The project follows this workflow:

1. Load and understand the dataset
2. Check and handle missing values
3. Perform Exploratory Data Analysis (EDA)
4. Select features and target variable
5. Split the data into training and testing sets
6. Apply One-Hot Encoding to categorical features
7. Train multiple Machine Learning models
8. Evaluate and compare model performance
9. Check model generalization
10. Select the best model
11. Predict delivery time for new unseen data
12. Save the trained model and encoder
13. Use the saved model in a terminal-based prediction application

---

## Exploratory Data Analysis

The analysis showed several important patterns:

- Distance had the strongest relationship with delivery time.
- Preparation time had a positive relationship with delivery time.
- Higher traffic levels generally resulted in longer delivery times.
- Weather conditions affected average delivery time.
- Courier experience showed a small negative relationship with delivery time.
- Vehicle type and time of day had relatively smaller effects.

---

## Models Tested

The following Machine Learning models were trained and evaluated:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 5.90 | 8.82 | **0.826** |
| Decision Tree | 10.65 | 15.33 | 0.476 |
| Random Forest | 6.82 | 9.70 | 0.790 |
| Gradient Boosting | 6.32 | 9.11 | 0.815 |

### Final Model

**Linear Regression** was selected as the final model because it achieved the best overall performance on the test data.

- **MAE:** 5.90 minutes
- **RMSE:** 8.82 minutes
- **R² Score:** 0.826

The model showed good generalization and outperformed the other tested models on the test set.

---

## Model Interpretation

The Linear Regression model learned meaningful relationships between the input features and delivery time.

Some key observations:

- Increasing distance generally increases predicted delivery time.
- Increasing preparation time generally increases predicted delivery time.
- Higher courier experience tends to slightly reduce predicted delivery time.
- High traffic and difficult weather conditions can increase predicted delivery time.

---

## Predicting New Data

The trained model can predict delivery time for completely new inputs.

Example:

```text
Distance: 10 km
Weather: Rainy
Traffic: High
Time of Day: Evening
Vehicle: Bike
Preparation Time: 20 minutes
Courier Experience: 3 years
```

Example prediction:

```text
Estimated Delivery Time: approximately 70.37 minutes
```

---

## Terminal Application

A basic terminal-based application was created using the saved model and encoder.

Run the application:

```bash
python app.py
```

The application asks the user for delivery-related details and returns an estimated delivery time.

Basic input validation was also added for categorical and numerical inputs.

---

## Project Structure

```text
Delivery_Time_Prediction/
│
├── data/
│   └── delivery_data.csv
│
├── notebooks/
│   └── delivery_time_prediction.ipynb
│
├── models/
│   ├── delivery_time_model.pkl
│   └── encoder.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- SciPy
- Joblib
- Matplotlib
- Jupyter Notebook

---

## Future Improvements

Possible future improvements include:

- Building a web interface with dropdown inputs
- Adding more real-world delivery features
- Testing the model on a larger dataset
- Improving input validation using training-data ranges
- Experimenting with additional feature engineering and models

---

## Author

Dattu