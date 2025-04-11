# -*- coding: utf-8 -*-

import joblib
from tensorflow.keras.models import load_model

for name, model in models.items():
    joblib.dump(model, f"{name}_model.pkl")
    print(f"{name} model saved as {name}_model.pkl")

def predict_with_model(model_name, input_data):
    if model_name in ["Deep Learning", "LSTM"]:
        # Load the Keras model

        print(f"{model_name.lower().replace(' ', '_')}_model.h5")
        # model = load_model(f"{model_name.lower().replace(' ', '_')}_model.h5")
        model = load_model(f"{model_name.lower().replace(' ', '_')}_model.h5", compile=False)


        # For LSTM, reshape input data to match LSTM's required input shape
        if model_name == "LSTM":
            input_data_scaled = scaler.transform([input_data])
            input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))
        else:
            input_data_scaled = scaler.transform([input_data])

        prediction = model.predict(input_data_scaled)
        return prediction[0]

    else:
        # Load the traditional ML model using joblib
        model = joblib.load(f"{model_name}_model.pkl")

        input_data_scaled = scaler.transform([input_data])
        prediction = model.predict(input_data_scaled)
        return prediction[0]

# Prediction function
print("Select a model to use for prediction:")
print("1. Linear Regression\n2. Random Forest\n3. Gradient Boosting\n4. SVR\n5. XGBoost\n6. LightGBM\n7. Deep Learning\n8. LSTM")
selected_model = input("Enter the model name (Linear Regression, Random Forest, Gradient Boosting, SVR, XGBoost, LightGBM, Deep Learning, LSTM): ")

input_data = [
    2,  # region
    1,  # type
    150,  # area_sq_km
    30,  # damage_percent
    1,  # type_of_damage
    2,  # new_type
    50,  # budget_million_usd
    3,  # recovery_priority
    365, # duration
    1   # funding_source
]

predicted_value = predict_with_model(selected_model, input_data)
print(f"Predicted success rate: {predicted_value:.2f}%")