from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from scipy.optimize import differential_evolution

# =====================================================================
# STEP 1: WIND POWER SIMULATION SETTINGS & API FETCH
# =====================================================================
start_date = "2025-05-24"
num_days = 30
LAT = 9.0503
LON = 79.7869
swept_area = 400000  # m²
air_density = 1.225
target_power_W = 100e6  # 100 MW reference

# Calculate correct end date
start_dt = datetime.strptime(start_date, "%Y-%m-%d")
end_dt = start_dt + timedelta(days=num_days - 1)
end_date = end_dt.strftime("%Y-%m-%d")

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "wind_speed_10m",
    "timezone": "Asia/Colombo",
}

print(f"Fetching data from {start_date} to {end_date}...")
response = requests.get(url, params=params)

if response.status_code != 200:
    print(f"Failed to fetch data: {response.status_code}")
    exit()

data = response.json()
if "hourly" not in data:
    print("No hourly data found in response.")
    exit()

# =====================================================================
# STEP 2: WIND DATA PROCESSING
# =====================================================================
df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date

df["wind_speed_ms"] = df["wind_speed_10m"] / 3.6
df["power_density"] = 0.5 * air_density * (df["wind_speed_ms"] ** 3)

daily_group = df.groupby("date")
wind_results = []

for date, group in daily_group:
    mean_power_density = group["power_density"].mean()
    total_power = mean_power_density * swept_area
    power_diff = total_power - target_power_W
    green_hydrogen_kg = power_diff / (50 * 10**3) if power_diff > 0 else 0
    efficient_power_MW = (power_diff * 0.35) / 10**6

    wind_results.append(
        {
            "date": str(date),
            "mean_power_density_Wm2": mean_power_density,
            "total_power_MW": total_power / 1e6,
            "power_difference_MW": power_diff / 1e6,
            "green_hydrogen_kg": green_hydrogen_kg,
            "efficient_power_MW": efficient_power_MW,
        }
    )

result_df = pd.DataFrame(wind_results)

print("\n===== Initial Wind Simulation Completed =====")
print(f"Average Power Density: {result_df['mean_power_density_Wm2'].mean():.2f} W/m²")
print(f"Average Turbine Power: {result_df['total_power_MW'].mean():.2f} MW\n")


# =====================================================================
# STEP 3: ELECTROLYZER OPTIMIZATION MODEL SETUP
# =====================================================================
P_max_single = 10.0  # Maximum operational limit per electrolyzer (MW)
num_units = 5  # Total available physical assets
bounds = [(0.0, P_max_single)] * num_units


def single_electrolyzer_efficiency(P):
    """Calculates non-linear efficiency profile of a single 5MW unit."""
    P = np.array(P, dtype=float)
    eff = 0.55 + 0.18 * (1 - np.exp(-2 * P / 10.0)) - 0.02 * (P / 10.0) ** 2
    return np.where(P < 0.01, 0.0, eff)


def optimize_plant_loading(power_allocations, P_target):
    """Fitness function evaluating the H2 output versus physics constraints."""
    power_allocations = np.array(power_allocations)
    efficiencies = single_electrolyzer_efficiency(power_allocations)
    total_h2_output = np.sum(power_allocations * efficiencies)

    power_error = abs(np.sum(power_allocations) - P_target)
    penalty = 1e6 * (power_error**2)

    return -total_h2_output + penalty


# =====================================================================
# STEP 4: SEQUENTIAL RUN OVER THE 30-DAY TIMELINE
# =====================================================================
print("=" * 140)
print("RUNNING DISPATCH OPTIMIZATION ENGINE FOR ACTIVE GENERATION SURPLUS DAYS")
print("=" * 140)

final_rows = []

for idx, row in result_df.iterrows():
    p_target = row["power_difference_MW"]
    date_str = row["date"]

    # Base dictionary setup for the final table entry
    report_entry = {
        "Date": date_str,
        "Input_MW": round(max(0, p_target), 2),
        "E1_MW": 0.0, "E2_MW": 0.0, "E3_MW": 0.0, "E4_MW": 0.0, "E5_MW": 0.0,
        "E1_%": 0.0, "E2_%": 0.0, "E3_%": 0.0, "E4_%": 0.0, "E5_%": 0.0,
        "Plant_%": 0.0,
    }

    # Only process optimization if input efficient power is greater than 0
    if p_target > 0:
        p_target_capped = min(p_target, P_max_single * num_units)

        result = differential_evolution(
            optimize_plant_loading,
            bounds,
            args=(p_target_capped,),
            strategy="best1bin",
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
        )

        optimized_allocations = np.where(result.x < 0.01, 0.0, result.x)
        individual_efficiencies = single_electrolyzer_efficiency(
            optimized_allocations
        )

        total_h2_power_out = np.sum(
            optimized_allocations * individual_efficiencies
        )
        global_plant_efficiency = (total_h2_power_out / p_target_capped) * 100

        # Map values to short-titled dictionary keys
        for i in range(num_units):
            report_entry[f"E{i+1}_MW"] = round(optimized_allocations[i], 2)
            report_entry[f"E{i+1}_%"] = round(individual_efficiencies[i] * 100, 1)

        report_entry["Plant_%"] = round(global_plant_efficiency, 1)
        report_entry["Input_MW"] = round(p_target_capped, 2)

    final_rows.append(report_entry)

# =====================================================================
# STEP 5: DISPLAY MULTI-COLUMN RESULTS DISPATCH SUMMARY (Fixed Display Settings)
# =====================================================================
dispatch_summary_df = pd.DataFrame(final_rows)

# Critical Override Settings: Stops Pandas from wrapping columns down to a new row block
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 2000)
pd.set_option("display.expand_frame_repr", False)

print("\n" + "=" * 140)
print("                                          FINAL OPTIMIZED DAILY DISPATCH CHRONOLOGY")
print("=" * 140)
print(dispatch_summary_df.to_string(index=False))
print("=" * 140)
