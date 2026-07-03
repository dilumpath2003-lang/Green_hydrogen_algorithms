import numpy as np
import pandas as pd
import requests

# =====================================================
# 1. FETCH LIVE DATA FROM OPEN-METEO API
# =====================================================
date = "2026-05-24"
LAT = 9.0503
LON = 79.7869
air_density = 1.225
swept_area = 400000

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": date,
    "end_date": date,
    "hourly": "wind_speed_10m",
    "timezone": "Asia/Colombo",
}

print("Fetching historical wind data from API...")
response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame(data["hourly"])
df["wind_speed_ms"] = df["wind_speed_10m"] / 3.6
df["power_density"] = 0.5 * air_density * (df["wind_speed_ms"] ** 3)

P_target = (df["power_density"].mean() * swept_area) / 1e6

# =====================================================
# 2. REFERENCE POWER & EQUAL DISTRIBUTION
# =====================================================
P_ref = 100.0
num_units = 5
P_max_single = 10.0

# Calculate total power allocated for hydrogen production
P_difference = abs(P_target - P_ref)

# Industry Standard: Distribute total power equally among all units
P_for_one = P_difference / num_units


# =====================================================
# 3. EFFICIENCY FUNCTION
# =====================================================
def single_electrolyzer_efficiency(P):
    P = np.array(P, dtype=float)
    eff = 0.55 + 0.18 * (1 - np.exp(-2 * P / 10.0)) - 0.02 * (P / 10.0) ** 2
    return np.where(P < 0.01, 0.0, eff)


# =====================================================
# 4. EVALUATE OBTAINED POWER EFFICIENCY
# =====================================================
# Calculate efficiency for a single unit based on its obtained power
efficiency_per_unit = single_electrolyzer_efficiency(P_for_one)

# Since power is split equally, system efficiency equals single unit efficiency
system_efficiency = efficiency_per_unit

print("\n--- Electrolyzer Performance Report ---")
print(f"Total Available Power (P_difference): {P_difference:.2f} MW")
print(f"Number of Active Electrolyzers:       {num_units}")
print(f"Obtained Power per Electrolyzer:      {P_for_one:.2f} MW")
print(f"Individual Unit Efficiency:           {efficiency_per_unit * 100:.2f}%")
print(f"Overall System Efficiency:            {system_efficiency * 100:.2f}%")
