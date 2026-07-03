import random
import numpy as np
import requests
import pandas as pd
from deap import base, creator, tools

# =====================================================
# STEP 0: FETCH DATA FROM API
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

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame(data["hourly"])
df["wind_speed_ms"] = df["wind_speed_10m"] / 3.6
df["power_density"] = 0.5 * air_density * (df["wind_speed_ms"] ** 3)

P_target = (df["power_density"].mean() * swept_area) / 1e6

print(f"API DATE: {date}")
print(f"Total Wind Power = {P_target:.2f} MW")

# =====================================================
# THRESHOLD LOGIC
# =====================================================
THRESHOLD = 100.0

if P_target > THRESHOLD:
    P_electrolyzer_target = P_target - THRESHOLD
else:
    P_electrolyzer_target = P_target

# =====================================================
# PARAMETERS
# =====================================================
num_units = 5
P_max_single = 10.0
P_min_single = 0.5   # IMPORTANT: no zero allowed

max_cap = num_units * P_max_single
min_cap = num_units * P_min_single

if P_electrolyzer_target > max_cap:
    P_electrolyzer_target = max_cap

if P_electrolyzer_target < min_cap:
    raise ValueError("Target too low for minimum constraint")

# =====================================================
# EFFICIENCY MODEL
# =====================================================
def efficiency(P):
    P = np.array(P)
    eff = 0.55 + 0.18*(1 - np.exp(-2*P/10)) - 0.02*(P/10)**2
    return np.where(P < 0.01, 0, eff)

# =====================================================
# REPAIR FUNCTION (CORE LOGIC)
# =====================================================
def repair(ind):
    x = np.array(ind, dtype=float)

    # enforce bounds
    x = np.clip(x, P_min_single, P_max_single)

    # force sum constraint
    total = np.sum(x)
    diff = P_electrolyzer_target - total

    # iterative correction
    for _ in range(10):
        if abs(diff) < 1e-6:
            break

        free = np.where(
            (x > P_min_single + 1e-6) &
            (x < P_max_single - 1e-6)
        )[0]

        if len(free) == 0:
            break

        x[free] += diff / len(free)
        x = np.clip(x, P_min_single, P_max_single)

        diff = P_electrolyzer_target - np.sum(x)

    return creator.Individual(x.tolist())

# =====================================================
# FITNESS FUNCTION
# =====================================================
def evaluate(ind):
    ind = np.array(ind)
    eff = efficiency(ind)
    return (np.sum(ind * eff),)

# =====================================================
# DEAP SETUP
# =====================================================
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except:
    pass

toolbox = base.Toolbox()

# random valid individual
def create_ind():
    base_val = P_min_single
    remaining = P_electrolyzer_target - num_units * P_min_single

    w = np.random.rand(num_units)
    w = w / np.sum(w)

    ind = base_val + w * remaining
    return creator.Individual(ind.tolist())

toolbox.register("individual", create_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# =====================================================
# INITIAL POPULATION
# =====================================================
pop = toolbox.population(n=20)

# =====================================================
# EVOLUTION LOOP
# =====================================================
for gen in range(20):

    print("\n" + "=" * 90)
    print(f"GENERATION {gen+1}")
    print("=" * 90)

    # evaluate + repair
    for i in range(len(pop)):
        pop[i] = repair(pop[i])
        pop[i].fitness.values = toolbox.evaluate(pop[i])

    # PRINT ALL 20 INDIVIDUALS
    for idx, ind in enumerate(pop, start=1):
        eff = efficiency(ind)
        print(
            f"Ind {idx:02d} | "
            f"E1={ind[0]:.2f} "
            f"E2={ind[1]:.2f} "
            f"E3={ind[2]:.2f} "
            f"E4={ind[3]:.2f} "
            f"E5={ind[4]:.2f} | "
            f"Total={sum(ind):.2f} | "
            f"Fitness={ind.fitness.values[0]:.2f}"
        )

    # selection
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # crossover
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7:
            toolbox.mate(c1, c2)
            c1[:] = repair(c1)
            c2[:] = repair(c2)
            del c1.fitness.values
            del c2.fitness.values

    # mutation
    for ind in offspring:
        if random.random() < 0.2:
            toolbox.mutate(ind)
            ind[:] = repair(ind)
            del ind.fitness.values

    pop = offspring
# =====================================================
# FINAL RESULT
# =====================================================
best = tools.selBest(pop, 1)[0]
best_eff = efficiency(best)

print("\n================ FINAL RESULT ================")
for i in range(num_units):
    print(f"E{i+1}: {best[i]:.2f} MW | Eff: {best_eff[i]*100:.2f}%")

print("\nTotal Power:", sum(best))
print("Target Power:", P_electrolyzer_target)
print("Fitness:", best.fitness.values[0])