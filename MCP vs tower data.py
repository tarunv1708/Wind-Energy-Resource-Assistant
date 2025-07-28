import pandas as pd
import numpy as np
from scipy.stats import weibull_min, vonmises, entropy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ========================= Step 1: Load and Process Data =========================

# Load GridMET Data
gridmet_data = pd.read_csv(r"E:\3rd SEM\Wind Energy\Final Project\gridmet3_output.csv")
gridmet_data.rename(columns={"ws": "ws_gridmet", "wd": "wd_gridmet", "day": "datetime"}, inplace=True)
gridmet_data["datetime"] = pd.to_datetime(gridmet_data["datetime"])

# Resample GridMET Data to Hourly Resolution
gridmet_data.set_index("datetime", inplace=True)
gridmet_hourly = gridmet_data.resample("h").interpolate()
gridmet_hourly["wd_gridmet"] = gridmet_data["wd_gridmet"].resample("h").interpolate(method="nearest")
gridmet_hourly["u_gridmet"] = -gridmet_hourly["ws_gridmet"] * np.sin(np.radians(gridmet_hourly["wd_gridmet"]))
gridmet_hourly["v_gridmet"] = -gridmet_hourly["ws_gridmet"] * np.cos(np.radians(gridmet_hourly["wd_gridmet"]))

# Load Tower Data
tower_data = pd.read_csv(r"E:\3rd SEM\Wind Energy\Final Project\HW1_2024_TowerData.csv")
tower_data.rename(columns={"Actual Dates and Times": "datetime", "WindSpeed (50.2 m)": "ws_tower", "Direction (Deg)": "wd_tower"}, inplace=True)
tower_data["datetime"] = pd.to_datetime(tower_data["datetime"])

# Merge Tower and GridMET Data
merged_data = pd.merge(tower_data, gridmet_hourly, on="datetime", how="inner")

# ========================= Step 2: MCP Process =========================

# Wind Speed Regression
X = merged_data["ws_gridmet"].values.reshape(-1, 1)
y = merged_data["ws_tower"].values
ws_model = LinearRegression().fit(X, y)

# Predict Long-Term Wind Speeds
gridmet_hourly["ws_tower_predicted"] = ws_model.predict(gridmet_hourly["ws_gridmet"].values.reshape(-1, 1))

# Add Residual Uncertainty
residual_std = np.std(y - ws_model.predict(X))
gridmet_hourly["ws_tower_with_uncertainty"] = gridmet_hourly["ws_tower_predicted"] + np.random.normal(0, residual_std, len(gridmet_hourly))

# Wind Direction MCP (u and v components)
merged_data["u_tower"] = -merged_data["ws_tower"] * np.sin(np.radians(merged_data["wd_tower"]))
merged_data["v_tower"] = -merged_data["ws_tower"] * np.cos(np.radians(merged_data["wd_tower"]))

u_model = LinearRegression().fit(merged_data[["u_gridmet"]], merged_data["u_tower"])
v_model = LinearRegression().fit(merged_data[["v_gridmet"]], merged_data["v_tower"])

gridmet_hourly["u_tower_predicted"] = u_model.predict(gridmet_hourly[["u_gridmet"]])
gridmet_hourly["v_tower_predicted"] = v_model.predict(gridmet_hourly[["v_gridmet"]])
gridmet_hourly["wd_tower_predicted"] = np.degrees(
    np.arctan2(-gridmet_hourly["u_tower_predicted"], -gridmet_hourly["v_tower_predicted"])
) % 360

# ========================= Step 3: Weibull Distribution =========================

# Fit Weibull Distributions
tower_speeds = merged_data["ws_tower"].dropna()
mcp_speeds = gridmet_hourly["ws_tower_predicted"].dropna()

shape_tower, loc_tower, scale_tower = weibull_min.fit(tower_speeds, floc=0)
shape_mcp, loc_mcp, scale_mcp = weibull_min.fit(mcp_speeds, floc=0)

# Compare Distributions (KL Divergence)
wind_speed_range = np.linspace(0, max(tower_speeds.max(), mcp_speeds.max()), 100)
tower_pdf = weibull_min.pdf(wind_speed_range, shape_tower, loc_tower, scale_tower)
mcp_pdf = weibull_min.pdf(wind_speed_range, shape_mcp, loc_mcp, scale_mcp)
kl_div = entropy(tower_pdf, mcp_pdf)
print(f"KL Divergence between Tower and MCP Weibull Distributions: {kl_div:.4f}")

# Plot Weibull Distributions
plt.plot(wind_speed_range, tower_pdf, label="Tower Data", color="blue")
plt.plot(wind_speed_range, mcp_pdf, label="MCP Data", color="red")
plt.legend()
plt.title("Weibull Distribution Comparison")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# ========================= Step 4: Wind Direction Uncertainty =========================

# Fit von Mises Distributions
kappa_tower = vonmises.fit(merged_data["wd_tower"])[0]
kappa_mcp = vonmises.fit(gridmet_hourly["wd_tower_predicted"])[0]
print(f"Kappa (Tower): {kappa_tower:.2f}")
print(f"Kappa (MCP): {kappa_mcp:.2f}")

# ========================= Step 5: AEP Calculation =========================

# Example Turbine Power Curve
wind_speed = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5]
power_output = [1.7, 15.3, 30.8, 53.7, 77.4, 106.2, 139.7, 171.4, 211.6, 248.6, 294.1, 378.8, 438.9, 496.4, 578.4, 629.8, 668, 742.4, 783.6, 801.3, 819.4, 831.7, 841.8, 849.6, 850.4, 851.5, 851.9, 851.9]  # kW
power_curve = interp1d(wind_speed, power_output, bounds_error=False, fill_value=0)

# AEP for Tower and MCP Data
tower_probabilities = tower_pdf * np.diff(wind_speed_range, prepend=0)
mcp_probabilities = mcp_pdf * np.diff(wind_speed_range, prepend=0)
power_generated = power_curve(wind_speed_range)

turbine_count = 10
hours_per_year = 8760
aep_tower = np.sum(tower_probabilities * power_generated) * turbine_count * hours_per_year / 1e6
aep_mcp = np.sum(mcp_probabilities * power_generated) * turbine_count * hours_per_year / 1e6

print(f"AEP (Tower Data): {aep_tower:.2f} GWh")
print(f"AEP (MCP Data): {aep_mcp:.2f} GWh")

# ========================= Step 6: Monte Carlo Simulation =========================

n_simulations = 1000
aep_simulations_tower = []
aep_simulations_mcp = []

for _ in range(n_simulations):
    shape_tower_simulated = np.random.normal(shape_tower, 0.05)
    scale_tower_simulated = np.random.normal(scale_tower, 0.05)
    tower_pdf_simulated = weibull_min.pdf(wind_speed_range, shape_tower_simulated, loc_tower, scale_tower_simulated)
    tower_probabilities_simulated = tower_pdf_simulated * np.diff(wind_speed_range, prepend=0)
    aep_simulated_tower = np.sum(tower_probabilities_simulated * power_generated) * turbine_count * hours_per_year / 1e6
    aep_simulations_tower.append(aep_simulated_tower)

    shape_mcp_simulated = np.random.normal(shape_mcp, 0.05)
    scale_mcp_simulated = np.random.normal(scale_mcp, 0.05)
    mcp_pdf_simulated = weibull_min.pdf(wind_speed_range, shape_mcp_simulated, loc_mcp, scale_mcp_simulated)
    mcp_probabilities_simulated = mcp_pdf_simulated * np.diff(wind_speed_range, prepend=0)
    aep_simulated_mcp = np.sum(mcp_probabilities_simulated * power_generated) * turbine_count * hours_per_year / 1e6
    aep_simulations_mcp.append(aep_simulated_mcp)

mean_aep_tower = np.mean(aep_simulations_tower)
mean_aep_mcp = np.mean(aep_simulations_mcp)

print(f"Mean AEP (Tower Data): {mean_aep_tower:.2f} GWh")
print(f"Mean AEP (MCP Data): {mean_aep_mcp:.2f} GWh")

# Plot AEP Distributions
plt.hist(aep_simulations_tower, bins=30, alpha=0.5, label="Tower AEP", color="blue")
plt.hist(aep_simulations_mcp, bins=30, alpha=0.5, label="MCP AEP", color="red")
plt.axvline(np.mean(aep_simulations_tower), color="blue", linestyle="--", label="Mean Tower AEP")
plt.axvline(np.mean(aep_simulations_mcp), color="red", linestyle="--", label="Mean MCP AEP")
plt.legend()
plt.title("AEP Distribution Comparison")
plt.xlabel("AEP (GWh)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
