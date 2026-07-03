import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# STEP 1: DEFINE THE UPDATED 10MW EFFICIENCY MODEL
# =====================================================================
def single_electrolyzer_efficiency(P):
    """Calculates non-linear efficiency profile of a single 10MW unit."""
    P = np.array(P, dtype=float)
    # The updated scale formula from your 30-day wind script
    eff = 0.55 + 0.18 * (1 - np.exp(-2 * P / 10.0)) - 0.02 * (P / 10.0) ** 2
    # Apply physical standby threshold (OFF below 0.01 MW)
    return np.where(P < 0.01, 0.0, eff)

# =====================================================================
# STEP 2: GENERATE DATAFRAME VECTORS
# =====================================================================
P_vals = np.linspace(0, 10.0, 500)
eff_vals = single_electrolyzer_efficiency(P_vals) * 100 

online_mask = P_vals >= 0.01
P_max_eff = 10.0
max_eff = single_electrolyzer_efficiency(P_max_eff) * 100

# =====================================================================
# STEP 3: PLOT CONSTRUCT
# =====================================================================
plt.figure(figsize=(10, 6))

# Plot the Online Operating Curve
plt.plot(P_vals[online_mask], eff_vals[online_mask], 
         label='Online Operating Curve', color='#0284c7', lw=2.5)

# Plot the Standby/OFF state dropping down to 0%
plt.plot([0, 0.01], [0, 0], color='#ef4444', lw=3, label='Standby State (OFF)')
plt.scatter(0, 0, color='#ef4444', s=50, zorder=5)

# Highlight updated Peak Efficiency Node
plt.scatter(P_max_eff, max_eff, color='darkblue', s=100, zorder=6,
            label=f'Absolute Peak Efficiency ({max_eff:.2f}% at {P_max_eff:.1f}MW)')

# Graph Labels & Cosmetics
plt.title('Single 10MW Electrolyzer Unit: Efficiency Curve', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Allocated Power Input, P (MW)', fontsize=11, labelpad=10)
plt.ylabel('Thermal Stack Efficiency, $\eta$ (%)', fontsize=11, labelpad=10)
plt.xlim(-0.5, 10.5)
plt.ylim(-5, 75)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower center', fontsize=10, frameon=True, shadow=True)

plt.tight_layout()
plt.show()
