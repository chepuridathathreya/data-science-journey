import joblib
import pandas as pd
from scipy.sparse import hstack
model = joblib.load("D:/dattu/Data Science/05_machinelearning/ML_Projects/1.Delevery_time_detection/Models/modeldelivery_time_model.pkl")
encoder = joblib.load("D:/dattu/Data Science/05_machinelearning/ML_Projects/1.Delevery_time_detection/Models/encoder.pkl")
def get_positive_number(prompt):
    while True:
        try:
            value = float(input(prompt))

            if value >= 0:
                return value
            else:
                print("Please enter a valid positive number.")

        except ValueError:
            print("Invalid input. Please enter a number.")
def get_choice(prompt, valid_choices):
    while True:
        value = input(prompt).strip().title()

        if value in valid_choices:
            return value
        else:
            print("Invalid input. Please choose from:", ", ".join(valid_choices))
def get_number_in_range(prompt, min_value, max_value):
    while True:
        try:
            value = float(input(prompt))

            if min_value <= value <= max_value:
                return value
            else:
                print(
                    f"Please enter a value between "
                    f"{min_value} and {max_value}."
                )

        except ValueError:
            print("Invalid input. Please enter a number.")
            
def predict_delivery_time(
    distance,
    weather,
    traffic,
    time_of_day,
    vehicle,
    preparation_time,
    experience
):

    categorical_cols = [
        "Weather",
        "Traffic_Level",
        "Time_of_Day",
        "Vehicle_Type"
    ]

    numerical_cols = [
        "Distance_km",
        "Preparation_Time_min",
        "Courier_Experience_yrs"
    ]

    new_data = pd.DataFrame({
        "Weather": [weather],
        "Traffic_Level": [traffic],
        "Time_of_Day": [time_of_day],
        "Vehicle_Type": [vehicle],
        "Distance_km": [distance],
        "Preparation_Time_min": [preparation_time],
        "Courier_Experience_yrs": [experience]
    })

    new_data_encoded = encoder.transform(
        new_data[categorical_cols]
    )

    new_data_final = hstack([
        new_data_encoded,
        new_data[numerical_cols].values
    ])

    prediction = model.predict(new_data_final)

    return prediction[0]


print("===== DELIVERY TIME PREDICTOR =====")
distance = get_number_in_range(
    "Enter distance in km: ",
    1,
    20
)

weather = get_choice(
    "Enter weather (Clear, Foggy, Rainy, Snowy, Windy): ",
    ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
)

traffic = get_choice(
    "Enter traffic level (Low, Medium, High): ",
    ["Low", "Medium", "High"]
)

time_of_day = get_choice(
    "Enter time of day (Morning, Afternoon, Evening, Night): ",
    ["Morning", "Afternoon", "Evening", "Night"]
)

vehicle = get_choice(
    "Enter vehicle type (Bike, Car, Scooter): ",
    ["Bike", "Car", "Scooter"]
)
preparation_time = get_positive_number("Enter preparation time in minutes: ")
experience = get_positive_number("Enter courier experience in years: ")

result = predict_delivery_time(
    distance,
    weather,
    traffic,
    time_of_day,
    vehicle,
    preparation_time,
    experience
)

print("\nEstimated Delivery Time:", round(result, 2), "minutes")