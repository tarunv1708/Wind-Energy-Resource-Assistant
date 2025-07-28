import pandas as pd
import numpy as np
from scipy.stats import norm, weibull_min
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from windrose import WindroseAxes

# File Path
file_path = r"E:\3rd SEM\Wind Energy\Assignments\HW 1\HW1_2024_TowerData.xls"

# Error Handling for File Loading
try:
    data = pd.read_excel(file_path, engine='xlrd')
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Convert the date column to datetime and set it as the index
data['Actual Dates and Times'] = pd.to_datetime(data['Actual Dates and Times'])
data.set_index('Actual Dates and Times', inplace=True)

# Interpolate missing values
data['WindSpeed (50.2 m)'] = data['WindSpeed (50.2 m)'].interpolate(method='linear')
data['Direction (Deg)'] = data['Direction (Deg)'].interpolate(method='linear')

# Data Cleaning - Remove invalid wind directions
data = data[(data['Direction (Deg)'] >= 0) & (data['Direction (Deg)'] <= 360)]

# Check for missing or extreme wind speeds
data = data[data['WindSpeed (50.2 m)'] > 0]  # Remove calm or invalid data

# Save the cleaned data
cleaned_file_path = r"E:\3rd SEM\Wind Energy\Assignments\HW 1\Cleaned_TowerData.xlsx"
data.to_excel(cleaned_file_path, engine='openpyxl')
print(f"Cleaned data saved to {cleaned_file_path}")

# Basic Statistics
mean_speed = data['WindSpeed (50.2 m)'].mean()
std_speed = data['WindSpeed (50.2 m)'].std()
print(f"Mean Wind Speed: {mean_speed:.2f} m/s")
print(f"Standard Deviation: {std_speed:.2f} m/s")

# Confidence Intervals for Wind Speed
confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)
sem_speed = std_speed / np.sqrt(len(data))  # Standard Error of the Mean
ci_lower = mean_speed - z_score * sem_speed
ci_upper = mean_speed + z_score * sem_speed
print(f"95% Confidence Interval for Wind Speed: ({ci_lower:.2f}, {ci_upper:.2f}) m/s")

# Bootstrap Sampling for Robust Confidence Intervals
bootstrapped_means = [data['WindSpeed (50.2 m)'].sample(frac=1, replace=True).mean() for _ in range(1000)]
ci_lower_bootstrap = np.percentile(bootstrapped_means, 2.5)
ci_upper_bootstrap = np.percentile(bootstrapped_means, 97.5)
print(f"Bootstrap 95% Confidence Interval: ({ci_lower_bootstrap:.2f}, {ci_upper_bootstrap:.2f}) m/s")

# Weibull Distribution Fitting
shape, loc, scale = weibull_min.fit(data['WindSpeed (50.2 m)'])
print(f"Weibull Shape: {shape:.2f}, Scale: {scale:.2f}")
simulated_speeds = weibull_min.rvs(shape, loc, scale, size=5000)

# Turbine Power Curve (Interpolated)
wind_speed = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5]
power_output = [1.7, 15.3, 30.8, 53.7, 77.4, 106.2, 139.7, 171.4, 211.6, 248.6, 294.1, 378.8, 438.9, 496.4, 578.4, 629.8, 668, 742.4, 783.6, 801.3, 819.4, 831.7, 841.8, 849.6, 850.4, 851.5, 851.9, 851.9]
power_curve_interpolated = interp1d(wind_speed, power_output, bounds_error=False, fill_value=0)

# Apply power curve to simulated speeds
simulated_energy = power_curve_interpolated(simulated_speeds)

# Annual Energy Production (AEP)
turbine_count = 10
hours_per_year = 8760
aep_mean = np.mean(simulated_energy) * turbine_count * hours_per_year
aep_std = np.std(simulated_energy) * turbine_count * hours_per_year
print(f"Expected AEP: {aep_mean:.2f} kWh")
print(f"AEP Uncertainty (\u00b11 Std Dev): \u00b1{aep_std:.2f} kWh")

# Visualizations
# Wind Speed Histogram
plt.hist(simulated_speeds, bins=30, color='skyblue', edgecolor='black')
plt.title("Simulated Wind Speeds (Weibull Distribution)")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# AEP Uncertainty Visualization (in GWh)
aep_mean_gwh = aep_mean / 1e6
aep_std_gwh = aep_std / 1e6
plt.hist(simulated_energy * turbine_count * hours_per_year / 1e6, bins=30, color='orange', edgecolor='black')
plt.axvline(aep_mean_gwh, color='blue', linestyle='--', label=f'Mean AEP: {aep_mean_gwh:.2f} GWh')
plt.axvline(aep_mean_gwh - aep_std_gwh, color='red', linestyle='--', label=f'-1 Std Dev: {aep_mean_gwh - aep_std_gwh:.2f} GWh')
plt.axvline(aep_mean_gwh + aep_std_gwh, color='green', linestyle='--', label=f'+1 Std Dev: {aep_mean_gwh + aep_std_gwh:.2f} GWh')
plt.legend(fontsize=12)  # Increase font size for legend
plt.title("AEP Uncertainty Visualization")
plt.xlabel("AEP (GWh)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Wind Rose
ax = WindroseAxes.from_ax()
ax.bar(data['Direction (Deg)'], data['WindSpeed (50.2 m)'], bins=np.arange(0, 30, 5), normed=True, edgecolor='white')
ax.set_legend(fontsize=12)  # Increase font size for legend
plt.title("Wind Rose with Frequency Bins")
plt.show()
