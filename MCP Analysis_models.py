import pandas as pd
import numpy as np
from scipy.stats import weibull_min, entropy, norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ========================= Step 1: Load and Process Data =========================

# Load GridMET Data
gridmet_data = pd.read_csv(r"E:\3rd SEM\Wind Energy\Final Project\gridmet3_output.csv")
gridmet_data.rename(columns={"ws": "ws_gridmet", "wd": "wd_gridmet", "day": "datetime"}, inplace=True)
gridmet_data["datetime"] = pd.to_datetime(gridmet_data["datetime"])

# Resample GridMET Data to Hourly Resolution
gridmet_data.set_index("datetime", inplace=True)
gridmet_hourly = gridmet_data.resample("h").interpolate()

# Load Tower Data
tower_data = pd.read_csv(r"E:\3rd SEM\Wind Energy\Final Project\HW1_2024_TowerData.csv")
tower_data.rename(columns={"Actual Dates and Times": "datetime", "WindSpeed (50.2 m)": "ws_tower"}, inplace=True)
tower_data["datetime"] = pd.to_datetime(tower_data["datetime"])

# Merge Tower and GridMET Data
merged_data = pd.merge(tower_data, gridmet_hourly, on="datetime", how="inner")
X = merged_data["ws_gridmet"].values.reshape(-1, 1)
y = merged_data["ws_tower"].values

# ========================= Step 2: MCP Models =========================

# Linear Regression
linear_model = LinearRegression().fit(X, y)
pred_linear = linear_model.predict(X)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
pred_rf = rf_model.predict(X)

# Gaussian Process Regression (GPR) with Subsampling and Batch Processing
subsample_size = 5000
indices = np.random.choice(len(X), subsample_size, replace=False)
X_subsampled = X[indices]
y_subsampled = y[indices]

# Fit GPR on subsampled data
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gpr_model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    random_state=42,
    optimizer='fmin_l_bfgs_b',
    alpha=1e-10
)
gpr_model.fit(X_subsampled, y_subsampled)

# Batch processing for GPR predictions
batch_size = 1000
pred_gpr = []
for i in range(0, len(X), batch_size):
    batch = X[i:i + batch_size]
    pred_gpr.append(gpr_model.predict(batch))
pred_gpr = np.concatenate(pred_gpr)

# ========================= Step 3: Comparison Metrics =========================

# Variance of Residuals
variance_linear = np.var(y - pred_linear)
variance_rf = np.var(y - pred_rf)
variance_gpr = np.var(y - pred_gpr)

# Bias (Mean Difference)
bias_linear = np.mean(y - pred_linear)
bias_rf = np.mean(y - pred_rf)
bias_gpr = np.mean(y - pred_gpr)

# Print Variance and Bias
print("Variance and Bias of Predictions:")
print(f"Linear Regression - Variance: {variance_linear:.2f}, Bias: {bias_linear:.2f}")
print(f"Random Forest - Variance: {variance_rf:.2f}, Bias: {bias_rf:.2f}")
print(f"Gaussian Process - Variance: {variance_gpr:.2f}, Bias: {bias_gpr:.2f}")

# ========================= Step 4: Weibull Distribution =========================

# Fit Weibull Distributions
shape_tower, loc_tower, scale_tower = weibull_min.fit(y, floc=0)
shape_linear, loc_linear, scale_linear = weibull_min.fit(pred_linear, floc=0)
shape_rf, loc_rf, scale_rf = weibull_min.fit(pred_rf, floc=0)
shape_gpr, loc_gpr, scale_gpr = weibull_min.fit(pred_gpr, floc=0)

# KL Divergence
wind_speed_range = np.linspace(0, max(y.max(), pred_linear.max(), pred_rf.max(), pred_gpr.max()), 100)
tower_pdf = weibull_min.pdf(wind_speed_range, shape_tower, loc_tower, scale_tower)
linear_pdf = weibull_min.pdf(wind_speed_range, shape_linear, loc_linear, scale_linear)
rf_pdf = weibull_min.pdf(wind_speed_range, shape_rf, loc_rf, scale_rf)
gpr_pdf = weibull_min.pdf(wind_speed_range, shape_gpr, loc_gpr, scale_gpr)

kl_div_linear = entropy(tower_pdf, linear_pdf)
kl_div_rf = entropy(tower_pdf, rf_pdf)
kl_div_gpr = entropy(tower_pdf, gpr_pdf)

print("KL Divergence with Tower Data:")
print(f"Linear Regression: {kl_div_linear:.4f}")
print(f"Random Forest: {kl_div_rf:.4f}")
print(f"Gaussian Process: {kl_div_gpr:.4f}")

# ========================= Step 5: AEP Calculation =========================

# Turbine Power Curve (Interpolated)
wind_speed = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5]
power_output = [1.7, 15.3, 30.8, 53.7, 77.4, 106.2, 139.7, 171.4, 211.6, 248.6, 294.1, 378.8, 438.9, 496.4, 578.4, 629.8, 668, 742.4, 783.6, 801.3, 819.4, 831.7, 841.8, 849.6, 850.4, 851.5, 851.9, 851.9]
power_curve = interp1d(wind_speed, power_output, bounds_error=False, fill_value=0)

# Calculate AEP
power_generated = power_curve(wind_speed_range)
tower_probabilities = tower_pdf * np.diff(wind_speed_range, prepend=0)
linear_probabilities = linear_pdf * np.diff(wind_speed_range, prepend=0)
rf_probabilities = rf_pdf * np.diff(wind_speed_range, prepend=0)
gpr_probabilities = gpr_pdf * np.diff(wind_speed_range, prepend=0)

turbine_count = 10
hours_per_year = 8760
aep_tower = np.sum(tower_probabilities * power_generated) * turbine_count * hours_per_year / 1e6
aep_linear = np.sum(linear_probabilities * power_generated) * turbine_count * hours_per_year / 1e6
aep_rf = np.sum(rf_probabilities * power_generated) * turbine_count * hours_per_year / 1e6
aep_gpr = np.sum(gpr_probabilities * power_generated) * turbine_count * hours_per_year / 1e6

print("AEP Comparison (GWh):")
print(f"Tower Data: {aep_tower:.2f}")
print(f"Linear Regression: {aep_linear:.2f}")
print(f"Random Forest: {aep_rf:.2f}")
print(f"Gaussian Process: {aep_gpr:.2f}")

# ========================= Step 6: Visualize Results =========================

# Plot Weibull Distributions
plt.plot(wind_speed_range, tower_pdf, label="Tower Data", color="blue")
plt.plot(wind_speed_range, linear_pdf, label="Linear Regression", color="red")
plt.plot(wind_speed_range, rf_pdf, label="Random Forest", color="green")
plt.plot(wind_speed_range, gpr_pdf, label="Gaussian Process", color="purple")
plt.legend()
plt.title("Weibull Distribution Comparison")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# Plot AEP Comparison
models = ["Tower", "Linear", "Random Forest", "Gaussian Process"]
aep_values = [aep_tower, aep_linear, aep_rf, aep_gpr]
plt.bar(models, aep_values, color=["blue", "red", "green", "purple"])
plt.title("AEP Comparison")
plt.ylabel("AEP (GWh)")
plt.grid(axis="y")
plt.show()

# ========================= Step 7: AEP Distribution =========================

n_simulations = 1000
aep_simulations = {"Tower": [], "Linear": [], "Random Forest": [], "Gaussian Process": []}

for _ in range(n_simulations):
    # Tower Data Simulations
    shape_simulated = np.random.normal(shape_tower, 0.05)
    scale_simulated = np.random.normal(scale_tower, 0.05)
    tower_pdf_simulated = weibull_min.pdf(wind_speed_range, shape_simulated, loc_tower, scale_simulated)
    tower_probabilities_simulated = tower_pdf_simulated * np.diff(wind_speed_range, prepend=0)
    aep_simulated_tower = np.sum(tower_probabilities_simulated * power_generated) * turbine_count * hours_per_year / 1e6
    aep_simulations["Tower"].append(aep_simulated_tower)

    # Linear Regression Simulations
    shape_simulated = np.random.normal(shape_linear, 0.05)
    scale_simulated = np.random.normal(scale_linear, 0.05)
    linear_pdf_simulated = weibull_min.pdf(wind_speed_range, shape_simulated, loc_linear, scale_simulated)
    linear_probabilities_simulated = linear_pdf_simulated * np.diff(wind_speed_range, prepend=0)
    aep_simulated_linear = np.sum(linear_probabilities_simulated * power_generated) * turbine_count * hours_per_year / 1e6
    aep_simulations["Linear"].append(aep_simulated_linear)

    # Random Forest Simulations
    shape_simulated = np.random.normal(shape_rf, 0.05)
    scale_simulated = np.random.normal(scale_rf, 0.05)
    rf_pdf_simulated = weibull_min.pdf(wind_speed_range, shape_simulated, loc_rf, scale_simulated)
    rf_probabilities_simulated = rf_pdf_simulated * np.diff(wind_speed_range, prepend=0)
    aep_simulated_rf = np.sum(rf_probabilities_simulated * power_generated) * turbine_count * hours_per_year / 1e6
    aep_simulations["Random Forest"].append(aep_simulated_rf)

    # Gaussian Process Simulations
    shape_simulated = np.random.normal(shape_gpr, 0.05)
    scale_simulated = np.random.normal(scale_gpr, 0.05)
    gpr_pdf_simulated = weibull_min.pdf(wind_speed_range, shape_simulated, loc_gpr, scale_simulated)
    gpr_probabilities_simulated = gpr_pdf_simulated * np.diff(wind_speed_range, prepend=0)
    aep_simulated_gpr = np.sum(gpr_probabilities_simulated * power_generated) * turbine_count * hours_per_year / 1e6
    aep_simulations["Gaussian Process"].append(aep_simulated_gpr)

# Plot AEP Distributions
plt.figure(figsize=(10, 6))
for model, simulations in aep_simulations.items():
    plt.hist(simulations, bins=30, alpha=0.5, label=f"{model} AEP", edgecolor="black")
plt.title("AEP Distribution by Model")
plt.xlabel("AEP (GWh)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
