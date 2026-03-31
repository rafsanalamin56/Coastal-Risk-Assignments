"""
Wind Setup Calculation along a Coastal Transect
================================================
Computes wind-induced water level setup (S) along a cross-shore transect
for a range of wind conditions, using the depth-averaged momentum balance:

    dS/dx = (C * U_eff^2) / (g * D)

where:
    C   = bottom friction / wind stress coefficient  [dimensionless]
    U   = wind speed  [m/s]
    g   = gravitational acceleration  [m/s^2]
    D   = total water depth = initial depth + setup  [m]

Two transect profiles are compared:
    - Base transect     : original bathymetry
    - Shallow transect  : base with two nearshore points reduced by 1 m

Results are printed in comma-separated format for easy copy-paste into Excel.
"""

import numpy as np


# =============================================================================
# Physical constants
# =============================================================================

G = 9.81          # gravitational acceleration  [m/s^2]
C = 3.2e-6        # combined wind stress / drag coefficient  [-]

# =============================================================================
# Numerical settings
# =============================================================================

TOLERANCE      = 1e-11   # convergence criterion on setup at the shore [m]
MAX_ITERATIONS = 500     # safety cap on the iterative solver

# =============================================================================
# Transect bathymetry
# Columns: [distance from shore (km), initial water depth (m)]
# Distance = 0 is the open-sea boundary; distance = 11 km is the shoreline.
# =============================================================================

TRANSECT_BASE = np.array([
    [0,  15.0],
    [1,  15.0],
    [4,  10.0],
    [6,   5.0],
    [8,   2.8],
    [9,   1.9],
    [10,  1.4],
    [11,  0.0],
])

# Shallow variant: nearshore depths at index 5 and 6 reduced by 1 m
TRANSECT_SHALLOW = TRANSECT_BASE.copy()
TRANSECT_SHALLOW[5, 1] -= 1.0
TRANSECT_SHALLOW[6, 1] -= 1.0
TRANSECT_SHALLOW[:, 1]  = np.maximum(TRANSECT_SHALLOW[:, 1], 0.0)  # no negative depths

# =============================================================================
# Wind scenarios: (label, speed [m/s], angle relative to shore-normal [deg])
# Angle = 0   -> shore-normal (maximum effective component)
# Angle = 45  -> oblique
# Angle = 90  -> shore-parallel (zero setup contribution)
# =============================================================================

WIND_SCENARIOS = [
    ("Light breeze",   3.0,  0),
    ("Strong breeze", 11.0,  0),
    ("Strong breeze", 11.0, 45),
    ("Strong breeze", 11.0, 90),
    ("Gale",          19.0,  0),
    ("Hurricane",     35.0,  0),
    ("Hurricane",     35.0, 45),
]


# =============================================================================
# Solver
# =============================================================================

def compute_setup(transect, wind_speed, angle_deg):
    """
    Iteratively solve for wind setup S(x) along the transect.

    Parameters
    ----------
    transect   : ndarray, shape (n, 2)
        Columns are [distance_km, initial_depth_m].
    wind_speed : float
        10-m wind speed [m/s].
    angle_deg  : float
        Wind angle relative to shore-normal [degrees].
        Only the shore-normal component drives setup: U_eff = U * cos(angle).

    Returns
    -------
    S : ndarray, shape (n,)
        Wind setup [m] at each transect node.
    """
    angle_rad = np.radians(angle_deg)
    x_km      = transect[:, 0]
    D0        = transect[:, 1]
    dx_m      = np.diff(x_km) * 1000.0          # segment lengths in metres

    U_eff_sq  = (wind_speed * np.cos(angle_rad)) ** 2
    forcing   = (C * U_eff_sq) / G              # numerator of dS/dx

    S         = np.zeros(len(D0))
    S_prev_shore = 0.0

    for _ in range(MAX_ITERATIONS):
        S_new = np.zeros_like(S)

        for i in range(len(D0) - 1):
            total_depth = max(D0[i] + S[i], 1e-6)   # avoid division by zero
            dS_dx       = forcing / total_depth
            S_new[i+1]  = S[i] + dS_dx * dx_m[i]

        if abs(S_new[-1] - S_prev_shore) < TOLERANCE:
            return S_new

        S_prev_shore = S_new[-1]
        S = S_new

    # Return best estimate if convergence was not reached within MAX_ITERATIONS
    return S


# =============================================================================
# Output
# =============================================================================

def print_results(base_transect, shallow_transect, scenarios):
    """
    For each wind scenario, print setup profiles for both transects
    in comma-separated format (Distance_km, Setup_Base_m, Setup_Shallow_m).
    """
    x_km = base_transect[:, 0]

    for label, speed, angle in scenarios:
        S_base    = compute_setup(base_transect,    speed, angle)
        S_shallow = compute_setup(shallow_transect, speed, angle)

        print("\n" + "=" * 50)
        print(f"  {label}  |  U = {speed} m/s  |  angle = {angle}°")
        print("=" * 50)
        print("Distance_km, Setup_Base_m, Setup_Shallow_m")

        for i in range(len(x_km)):
            print(f"{x_km[i]:.2f}, {S_base[i]:.6f}, {S_shallow[i]:.6f}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print_results(TRANSECT_BASE, TRANSECT_SHALLOW, WIND_SCENARIOS)
