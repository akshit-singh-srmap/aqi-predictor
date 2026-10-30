
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# 1. LOAD DATASET

df = pd.read_csv(r"D:\singularity\beijing+multi+site+air+quality+data\PRSA2017_Data_20130301-20170228\PRSA_Data_20130301-20170228\PRSA_Data_Aotizhongxin_20130301-20170228.csv")

df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

df = df.sort_values("datetime").reset_index(drop=True)


# 2. VISUALIZE POLLUTION TREND

daily_pm25 = df.set_index("datetime")["PM2.5"].resample("D").mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_pm25)
plt.title("Daily PM2.5 Pollution Trend")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.grid()
plt.show()


# 3. FEATURE ENGINEERING

df["PM2.5_previous"] = df["PM2.5"].shift(1)

df["PM2.5_24h_avg"] = df["PM2.5"].rolling(24).mean()

df["target"] = df["PM2.5"].shift(-1)

features = [
    "PM2.5",
    "PM2.5_previous",
    "PM2.5_24h_avg",
    "PM10",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "hour",
    "month"
]

# Keep only consecutive hourly observations
valid = (
    (df["datetime"] - df["datetime"].shift(1) == pd.Timedelta(hours=1))
    &
    (df["datetime"].shift(-1) - df["datetime"] == pd.Timedelta(hours=1))
)

data = df[valid].dropna(subset=features + ["target"])

X = data[features]
y = data["target"]

print("Usable observations:", len(data))


# 4. CHRONOLOGICAL TRAIN/TEST SPLIT

split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# 5. LINEAR REGRESSION

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_prediction)
lr_r2 = r2_score(y_test, lr_prediction)


# 6. RANDOM FOREST

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_prediction = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_prediction)
rf_r2 = r2_score(y_test, rf_prediction)


# 7. MODEL COMPARISON

print("\nMODEL COMPARISON")

print("Linear Regression:", round(lr_mae, 2), round(lr_r2, 3))
print("Random Forest:", round(rf_mae, 2), round(rf_r2, 3))

if lr_mae <= rf_mae:
    best_model = lr
    prediction = lr_prediction
    print("\nSelected Model: Linear Regression")
else:
    best_model = rf
    prediction = rf_prediction
    print("\nSelected Model: Random Forest")


# 8. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(12, 5))

plt.plot(y_test.iloc[:200].values, label="Actual")
plt.plot(prediction[:200], label="Predicted")

plt.title("Actual vs Predicted PM2.5")
plt.xlabel("Hourly Observations")
plt.ylabel("PM2.5")
plt.legend()
plt.grid()
plt.show()


# 9. TERMINAL PREDICTION APPLICATION

print("\n NEXT-HOUR PM2.5 PREDICTOR ")

pm25 = float(input("Current PM2.5: "))
previous = float(input("Previous-hour PM2.5: "))
average = float(input("24-hour average PM2.5: "))

pm10 = float(input("PM10: "))
temp = float(input("Temperature: "))
pressure = float(input("Pressure: "))
dew = float(input("Dew point: "))
rain = float(input("Rainfall: "))
wind = float(input("Wind speed: "))

hour = int(input("Current hour (0-23): "))
month = int(input("Current month (1-12): "))


# Prepare input

new_data = pd.DataFrame([[
    pm25, previous, average, pm10, temp,
    pressure, dew, rain, wind, hour, month
]], columns=features)


# Predict

result = best_model.predict(new_data)[0]    

result = max(0, result)

print("\n PREDICTION RESULT ")
print(f"Next-hour PM2.5: {result:.2f} ug/m3")
