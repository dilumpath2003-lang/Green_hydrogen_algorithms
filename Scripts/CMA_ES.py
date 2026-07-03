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
# 2. REFERENCE POWER & DIFFERENCE
# =====================================================
P_ref = 100.0
num_units = 5

P_max_single = 10.0       # Maximum power per electrolyzer
P_min_single = 0.50       # Minimum power per electrolyzer (must not be zero)

P_difference = abs(P_target - P_ref)

# Check feasibility
if P_difference < num_units * P_min_single:
    raise ValueError(
        "Power difference is too small to satisfy the minimum power constraint."
    )

if P_difference > num_units * P_max_single:
    raise ValueError(
        "Power difference exceeds the maximum total electrolyzer capacity."
    )

# =====================================================
# 3. EFFICIENCY FUNCTION
# =====================================================
def single_electrolyzer_efficiency(P):
    P = np.array(P, dtype=float)
    eff = 0.55 + 0.18 * (1 - np.exp(-2 * P / 10.0)) - 0.02 * (P / 10.0) ** 2
    return np.where(P < 0.01, 0.0, eff)

# =====================================================
# 4. FITNESS FUNCTION
# =====================================================
def fitness(P):
    P = np.clip(P, P_min_single, P_max_single)
    eff = single_electrolyzer_efficiency(P)
    return np.sum(P * eff)

# =====================================================
# 5. SIMPLE CMA-ES STYLE OPTIMIZER
# =====================================================

population_size = 20
iterations = 100
sigma = 0.5

# Initial mean (all electrolyzers receive at least minimum power)
remaining = P_difference - num_units * P_min_single

weights = np.random.rand(num_units)
weights /= weights.sum()

mean = P_min_single + weights * remaining

# =====================================================
# Optimization Loop
# =====================================================

for generation in range(iterations):

    population = []

    for _ in range(population_size):

        noise = np.random.randn(num_units)

        candidate = mean + sigma * noise

        # Enforce minimum power
        candidate = np.maximum(candidate, P_min_single)

        # Remove minimum portion
        extra = candidate - P_min_single

        extra = np.maximum(extra, 0)

        # Redistribute remaining power
        if extra.sum() > 0:
            extra = extra / extra.sum() * remaining
        else:
            extra = np.ones(num_units) * (remaining / num_units)

        candidate = P_min_single + extra

        # Respect maximum limit
        candidate = np.minimum(candidate, P_max_single)

        # If clipping changed total power, redistribute again
        deficit = P_difference - candidate.sum()

        while abs(deficit) > 1e-6:

            available = np.where(candidate < P_max_single - 1e-6)[0]

            if len(available) == 0:
                break

            add = deficit / len(available)

            candidate[available] += add

            candidate = np.clip(candidate, P_min_single, P_max_single)

            deficit = P_difference - candidate.sum()

        population.append(candidate)

    population = np.array(population)

    # Evaluate population
    scores = np.array([fitness(ind) for ind in population])

    # Select best half
    best_idx = np.argsort(scores)[-population_size // 2 :]
    best = population[best_idx]

    # Update mean
    mean = np.mean(best, axis=0)

    # Reduce exploration
    sigma *= 0.99

# =====================================================
# FINAL RESULTS
# =====================================================

optimal_split = mean
optimal_eff = single_electrolyzer_efficiency(optimal_split)

print("\n" + "=" * 65)
print(f"API DATE LOADED              : {date}")
print(f"Wind Power Generated         : {P_target:.2f} MW")
print(f"Reference Power              : {P_ref:.2f} MW")
print(f"Net Power Difference         : {P_difference:.2f} MW")
print("=" * 65)
print("OPTIMAL CMA-ES POWER DISTRIBUTION")
print("-" * 65)

for i in range(num_units):
    print(
        f"Electrolyzer {i+1}: "
        f"{optimal_split[i]:6.2f} MW   "
        f"Efficiency = {optimal_eff[i]*100:6.2f}%"
    )

print("-" * 65)
print(f"Total Allocated Power        : {optimal_split.sum():.2f} MW")
print(f"Average Efficiency           : {optimal_eff.mean()*100:.2f}%")
print(f"Total Effective Power Score  : {fitness(optimal_split):.4f}")
print("=" * 65)