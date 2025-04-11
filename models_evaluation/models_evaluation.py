# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
import lightgbm as lgb

# Load dataset
df = pd.read_csv("recovery_data.csv")

# ML Pipeline
features = ["region", "type", "area_sq_km", "damage_percent", "type_of_damage", "new_type",
            "budget_million_usd", "recovery_priority", "duration", "funding_source"]
target = "success_rate_percent"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R² Score": r2}

# Deep Learning Model
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

dl_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

dl_model.fit(X_train_scaled, y_train,
             epochs=100,
             batch_size=32,
             validation_split=0.2,
             callbacks=callbacks,
             verbose=0)

y_pred_dl = dl_model.predict(X_test_scaled).flatten()
mse_dl = mean_squared_error(y_test, y_pred_dl)
mae_dl = mean_absolute_error(y_test, y_pred_dl)
rmse_dl = np.sqrt(mse_dl)
r2_dl = r2_score(y_test, y_pred_dl)
results["Deep Learning"] = {"MSE": mse_dl, "RMSE": rmse_dl, "MAE": mae_dl, "R² Score": r2_dl}

dl_model.save("deep_learning_model.h5")
print("Deep Learning model saved as deep_learning_model.h5")

# LSTM Model
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)

y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(y_test, y_pred_lstm)
results["LSTM"] = {"MSE": mse_lstm, "RMSE": rmse_lstm, "MAE": mae_lstm, "R² Score": r2_lstm}

lstm_model.save("lstm_model.h5")
print("LSTM model saved as lstm_model.h5")

results_df = pd.DataFrame(results).T
print(results_df)

plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y=results_df["R² Score"], palette="coolwarm")
plt.title("Model Performance - R² Score")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.show()