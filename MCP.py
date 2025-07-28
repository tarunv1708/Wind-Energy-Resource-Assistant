import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import weibull_min
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ===================== Step 1: MCP Process =====================

# Load GridMET Data
gridmet_data = pd.read_csv(r"E:\3rd SEM\Wind Energy\Final Project\gridmet3_output.csv")  # Replace with your GridMET file path
gridmet_data.rename(columns={"ws": "ws_gridmet", "wd": "wd_gridmet", "day": "datetime"}, inplace=True)
gridmet_data["datetime"] = pd.to_datetime(gridmet_data["datetime"])

# Resample GridMET Data to Hourly Resolution
gridmet_data.set_index("datetime", inplace=True)
gridmet_hourly = gridmet_data.resample("h").interpolate()
gridmet_hourly["wd_gridmet"] = gridmet_data["wd_gridmet"].resample("h").interpolate(method="nearest")
gridmet_hourly["u_gridmet"] = -gridmet_hourly["ws_gridmet"] * np.sin(np.radians(gridmet_hourly["wd_gridmet"]))
gridmet_hourly["v_gridmet"] = -gridmet_hourly["ws_gridmet"] * np.cos(np.radians(gridmet_hourly["wd_gridmet"]))

# Load Tower Data
tower_data = pd.read_csv(r"E:\3rd SEM\Wind Energy\Final Project\HW1_2024_TowerData.csv")  # Replace with your Tower file path
tower_data.rename(columns={"Actual Dates and Times": "datetime",
                           "WindSpeed (50.2 m)": "ws_tower",
                           "Direction (Deg)": "wd_tower"}, inplace=True)
tower_data["datetime"] = pd.to_datetime(tower_data["datetime"])

# Merge Tower and GridMET Data
merged_data = pd.merge(tower_data, gridmet_hourly, on="datetime", how="inner")

# Wind Speed Regression
X = merged_data["ws_gridmet"].values.reshape(-1, 1)
y = merged_data["ws_tower"].values
model = LinearRegression().fit(X, y)

# Predict Long-Term Wind Speeds
gridmet_hourly["ws_tower_predicted"] = model.predict(gridmet_hourly["ws_gridmet"].values.reshape(-1, 1))

# Add Residual Uncertainty
residual_std = np.std(y - model.predict(X))
gridmet_hourly["ws_tower_with_uncertainty"] = (
    gridmet_hourly["ws_tower_predicted"] +
    np.random.normal(0, residual_std, len(gridmet_hourly))
)

# Wind Direction MCP (u and v components)
merged_data["u_tower"] = -merged_data["ws_tower"] * np.sin(np.radians(merged_data["wd_tower"]))
merged_data["v_tower"] = -merged_data["ws_tower"] * np.cos(np.radians(merged_data["wd_tower"]))

# u and v Regression
u_model = LinearRegression().fit(merged_data[["u_gridmet"]], merged_data["u_tower"])
v_model = LinearRegression().fit(merged_data[["v_gridmet"]], merged_data["v_tower"])

# Predict Long-Term Wind Directions
gridmet_hourly["u_tower_predicted"] = u_model.predict(gridmet_hourly[["u_gridmet"]])
gridmet_hourly["v_tower_predicted"] = v_model.predict(gridmet_hourly[["v_gridmet"]])
gridmet_hourly["wd_tower_predicted"] = np.degrees(
    np.arctan2(-gridmet_hourly["u_tower_predicted"], -gridmet_hourly["v_tower_predicted"])
) % 360

# Save Processed Long-Term Data
gridmet_hourly.to_csv("processed_long_term_data2.csv", index=False)

# ===================== Step 2: Weibull Fitting =====================
wind_speeds = gridmet_hourly["ws_tower_predicted"].dropna()
shape, loc, scale = weibull_min.fit(wind_speeds, floc=0)
print(f"Weibull Shape (k): {shape:.2f}")
print(f"Weibull Scale (c): {scale:.2f}")

# Plot Weibull Distribution
x = np.linspace(0, wind_speeds.max(), 100)
weibull_pdf = weibull_min.pdf(x, shape, loc, scale)

plt.hist(wind_speeds, bins=30, density=True, alpha=0.6, color="blue", label="Wind Speed Data")
plt.plot(x, weibull_pdf, color="red", label=f"Weibull Fit (k={shape:.2f}, c={scale:.2f})")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Probability Density")
plt.title("Weibull Distribution Fit for Wind Speeds")
plt.legend()
plt.grid(True)
plt.show()

# ===================== Step 3: AEP Calculation =====================
# Example Turbine Power Curve
wind_speed = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5]
power_output = [1.7, 15.3, 30.8, 53.7, 77.4, 106.2, 139.7, 171.4, 211.6, 248.6, 294.1, 378.8, 438.9, 496.4, 578.4, 629.8, 668, 742.4, 783.6, 801.3, 819.4, 831.7, 841.8, 849.6, 850.4, 851.5, 851.9, 851.9]#Units in KW


# Interpolated Power Curve Function
power_curve = interp1d(wind_speed, power_output, bounds_error=False, fill_value=0)

# Weibull Probability for Wind Speeds
wind_speed_range = np.linspace(0, wind_speeds.max(), 100)
weibull_pdf = weibull_min.pdf(wind_speed_range, shape, loc, scale)

# Calculate AEP
probabilities = weibull_pdf * np.diff(wind_speed_range, prepend=0)
power_generated = power_curve(wind_speed_range)
turbine_count = 10
hours_per_year = 8760

aep = np.sum(probabilities * power_generated) * turbine_count * hours_per_year / 1e6
print(f"Annual Energy Production (AEP): {aep:.2f} GWh")

# ===================== Step 4: Monte Carlo Simulation for AEP Uncertainty =====================
n_simulations = 1000
aep_simulations = []

for _ in range(n_simulations):
    shape_simulated = np.random.normal(shape, 0.05)
    scale_simulated = np.random.normal(scale, 0.05)
    weibull_pdf_simulated = weibull_min.pdf(wind_speed_range, shape_simulated, loc, scale_simulated)
    probabilities_simulated = weibull_pdf_simulated * np.diff(wind_speed_range, prepend=0)
    aep_simulated = np.sum(probabilities_simulated * power_generated) * turbine_count * hours_per_year / 1e6
    aep_simulations.append(aep_simulated)

# Analyze AEP Distribution
aep_mean = np.mean(aep_simulations)
aep_std = np.std(aep_simulations)
print(f"AEP (Mean): {aep_mean:.2f} GWh ± {aep_std:.2f} GWh")

# Plot AEP Distribution
plt.hist(aep_simulations, bins=30, color="purple", edgecolor="black", alpha=0.7)
plt.axvline(aep_mean, color="red", linestyle="--", label=f"Mean AEP: {aep_mean:.2f} GWh")
plt.xlabel("AEP (GWh)")
plt.ylabel("Frequency")
plt.title("AEP Uncertainty Distribution")
plt.legend()
plt.grid(True)
plt.show()
