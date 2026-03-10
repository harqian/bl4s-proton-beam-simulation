import logging
import time

import matplotlib.pyplot as plt
import numpy as np

# ===================== PARAMETERS =====================

# -------- Beam --------
PROTON_KE_MEV = 7750.0        # kinetic energy in MeV (7.75 GeV beam)
PROTON_FLUX = 500000           # particles / (400ms spill) — rough estimate
BEAM_AREA = 2.0106192983e-10   # m^2, π(8µm)^2

# -------- Geometry --------
NUM_PHYSICAL_LAYERS = 4
PAIRS_PER_LAYER = 100          # water/bismuth pairs per physical layer
WATER_THICKNESS = 1e-5         # m per water sub-layer (10 µm)
BISMUTH_THICKNESS = 1e-5       # m per bismuth sub-layer (10 µm)

# -------- Boundary Conditions --------
H_COOLING = 1e3                # convective heat transfer coeff (W/m^2/K)
T_ENV = 293.0                  # K

# -------- Numerics --------
DX = 1e-5                      # spatial step (m) — match sub-layer thickness
SIMULATION_TIME = 3.0          # seconds

# ===================== CONSTANTS =====================

E_CHARGE = 1.602e-19           # C
M_E = 9.109e-31                # kg
C = 2.99792458e8               # m/s
PROTON_MASS_MEV = 938.272      # MeV/c^2

# Material properties
DENSITY_WATER = 1000.0;   CP_WATER = 4184.0;  K_WATER = 0.6
DENSITY_BISMUTH = 9780.0; CP_BISMUTH = 122.0;  K_BISMUTH = 7.9

I_WATER_EV = 75.0              # mean excitation energy (eV)
I_BISMUTH_EV = 823.0

Z_WATER = 7.42                 # effective Z
A_WATER = 14.84                # effective A (for electron density)
Z_BISMUTH = 83
A_BISMUTH = 208.98

THERMAL_EXPANSION_WATER = 2.07e-4  # 1/K
N_A = 6.022e23

logger = logging.getLogger(__name__)

# ===================== GEOMETRY =====================

def build_layers():
    """4 identical physical layers, each with PAIRS_PER_LAYER water/bismuth pairs."""
    layers = []
    for _ in range(NUM_PHYSICAL_LAYERS):
        for _ in range(PAIRS_PER_LAYER):
            layers.append(("water", WATER_THICKNESS))
            layers.append(("bismuth", BISMUTH_THICKNESS))
    return layers


def build_spatial_grid():
    layers = build_layers()
    z_positions = []
    material_map = []
    current_z = 0.0

    for mat, thickness in layers:
        n_cells = max(1, int(round(thickness / DX)))
        for _ in range(n_cells):
            z_positions.append(current_z)
            material_map.append(mat)
            current_z += DX

    return np.array(z_positions), np.array(material_map)


# ===================== BETHE-BLOCH (all in MeV) =====================

def bethe_bloch_MeV_per_m(KE_MeV, Z, A, I_eV, density):
    """
    Returns stopping power in MeV/m for a proton.
    Uses the standard PDG Bethe formula (no shell/density corrections).
    KE_MeV: proton kinetic energy in MeV
    """
    gamma = 1.0 + KE_MeV / PROTON_MASS_MEV
    beta2 = 1.0 - 1.0 / gamma**2
    beta2 = max(beta2, 1e-12)

    m_e_MeV = 0.511  # electron mass in MeV/c^2
    I_MeV = I_eV * 1e-6

    # Tmax for proton on electron (exact)
    Tmax = (2 * m_e_MeV * beta2 * gamma**2) / (
        1 + 2 * gamma * m_e_MeV / PROTON_MASS_MEV + (m_e_MeV / PROTON_MASS_MEV)**2
    )

    # electron density: n_e = N_A * Z * density / A (per m^3, density in kg/m^3 -> g/cm^3 conversion)
    density_gcc = density * 1e-3  # kg/m^3 -> g/cm^3
    n_e = N_A * Z / A * density_gcc  # per cm^3
    n_e *= 1e6  # per m^3

    # K = 4π r_e^2 m_e c^2 N_A = 0.3071 MeV cm^2 / mol (but we compute from n_e directly)
    r_e = 2.8179e-15  # classical electron radius in m
    prefactor = 4 * np.pi * r_e**2 * m_e_MeV * n_e  # MeV/m (after ×1/beta2 × log term)

    log_arg = 2 * m_e_MeV * beta2 * gamma**2 * Tmax / I_MeV**2
    if log_arg <= 0:
        return 0.0

    dEdx = prefactor / beta2 * (0.5 * np.log(log_arg) - beta2)  # MeV/m
    return max(dEdx, 0.0)


def stopping_power(KE_MeV, mat, density):
    if mat == "water":
        return bethe_bloch_MeV_per_m(KE_MeV, Z_WATER, A_WATER, I_WATER_EV, density)
    else:
        return bethe_bloch_MeV_per_m(KE_MeV, Z_BISMUTH, A_BISMUTH, I_BISMUTH_EV, density)


# ===================== PROTON TRANSPORT =====================

def transport_protons(KE0_MeV, material_map, T_profile):
    KE = KE0_MeV
    deposition_MeV = np.zeros(len(material_map))

    for i, mat in enumerate(material_map):
        if mat == "water":
            density = DENSITY_WATER * (1 - THERMAL_EXPANSION_WATER * (T_profile[i] - T_ENV))
        else:
            density = DENSITY_BISMUTH

        S = stopping_power(KE, mat, density)
        dE = S * DX  # MeV lost in this cell
        dE = min(dE, KE)

        KE -= dE
        deposition_MeV[i] = dE

        if KE <= 0:
            break

    return deposition_MeV, KE


# ===================== THERMAL SOLVER (implicit) =====================

def get_thermal_props(material_map):
    """Pre-compute arrays of rho, cp, k."""
    N = len(material_map)
    rho = np.empty(N); cp = np.empty(N); k = np.empty(N)
    for i, mat in enumerate(material_map):
        if mat == "water":
            rho[i] = DENSITY_WATER; cp[i] = CP_WATER; k[i] = K_WATER
        else:
            rho[i] = DENSITY_BISMUTH; cp[i] = CP_BISMUTH; k[i] = K_BISMUTH
    return rho, cp, k


def thermal_step_implicit(T, Q, rho, cp, k, dt):
    """
    Crank-Nicolson (tridiagonal solve) for 1D heat equation.
    Convective cooling at boundaries only.
    """
    N = len(T)
    alpha = k / (rho * cp)
    r = alpha * dt / (2 * DX**2)  # CN uses half-step

    # Build tridiagonal system: A @ T_new = rhs
    a = np.zeros(N)  # lower diagonal
    b = np.zeros(N)  # main diagonal
    c_diag = np.zeros(N)  # upper diagonal
    d = np.zeros(N)  # RHS

    for i in range(1, N-1):
        a[i] = -r[i]
        b[i] = 1 + 2*r[i]
        c_diag[i] = -r[i]
        d[i] = (r[i] * T[i-1] + (1 - 2*r[i]) * T[i] + r[i] * T[i+1]
                + dt * Q[i] / (rho[i] * cp[i]))

    # Left boundary: convective cooling
    h_coeff_L = H_COOLING * dt / (rho[0] * cp[0] * DX)
    b[0] = 1 + h_coeff_L
    c_diag[0] = 0
    d[0] = T[0] + h_coeff_L * T_ENV + dt * Q[0] / (rho[0] * cp[0])

    # Right boundary: convective cooling
    h_coeff_R = H_COOLING * dt / (rho[-1] * cp[-1] * DX)
    a[-1] = 0
    b[-1] = 1 + h_coeff_R
    d[-1] = T[-1] + h_coeff_R * T_ENV + dt * Q[-1] / (rho[-1] * cp[-1])

    # Thomas algorithm (tridiagonal solve)
    T_new = thomas_solve(a, b, c_diag, d)
    return T_new


def thomas_solve(a, b, c, d):
    N = len(d)
    c_ = np.zeros(N); d_ = np.zeros(N)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, N):
        m = a[i] / (b[i] - a[i] * c_[i-1])
        c_[i] = c[i] / (b[i] - a[i] * c_[i-1])
        d_[i] = (d[i] - a[i] * d_[i-1]) / (b[i] - a[i] * c_[i-1])
    x = np.zeros(N)
    x[-1] = d_[-1]
    for i in range(N-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]
    return x


# ===================== SIMULATION =====================

def compute_stable_dt(rho, cp, k):
    """Compute max stable dt for explicit method (informational), use for CN step sizing."""
    alpha_max = np.max(k / (rho * cp))
    dt_explicit_max = 0.5 * DX**2 / alpha_max
    # CN is unconditionally stable, but use a reasonable dt for accuracy
    dt = min(0.01, 10 * dt_explicit_max)  # allow larger steps since implicit
    return dt


def run_simulation():
    sim_start = time.perf_counter()

    logger.info("building spatial grid")
    z, material_map = build_spatial_grid()
    N = len(z)
    logger.info("grid ready: %s cells, total length %.2f mm", N, z[-1] * 1000)

    logger.info("precomputing thermal properties")
    rho, cp, k = get_thermal_props(material_map)
    dt = compute_stable_dt(rho, cp, k)
    logger.info("time step set to %.6f s", dt)

    T = np.full(N, T_ENV)  # start at ambient
    times = np.arange(0, SIMULATION_TIME, dt)
    num_steps = len(times)
    logger.info("running %s simulation steps over %.2f s", num_steps, SIMULATION_TIME)

    avg_temps = []
    exit_energies = []
    layer_avg_temps = []  # per-layer tracking

    # precompute layer boundaries
    cells_per_layer = N // NUM_PHYSICAL_LAYERS
    progress_interval = max(1, num_steps // 10)
    transport_total = 0.0
    thermal_total = 0.0

    for step_idx, t in enumerate(times, start=1):
        transport_start = time.perf_counter()
        deposition_MeV, KE_exit = transport_protons(PROTON_KE_MEV, material_map, T)
        transport_total += time.perf_counter() - transport_start

        # Convert MeV deposition to volumetric heat source (W/m^3)
        deposition_J = deposition_MeV * 1e6 * E_CHARGE  # MeV -> J
        Q = deposition_J * PROTON_FLUX / (BEAM_AREA)  # J/m (per unit area already in beam)
        # Actually: Q should be W/m^3 = (energy per particle per cell) * (particles/s) / (cell volume)
        # cell volume in 1D = DX * BEAM_AREA
        Q = deposition_J * PROTON_FLUX / (DX * BEAM_AREA) * BEAM_AREA  # simplifies to deposition_J * FLUX / DX
        Q = deposition_J * PROTON_FLUX / DX

        thermal_start = time.perf_counter()
        T = thermal_step_implicit(T, Q, rho, cp, k, dt)
        thermal_total += time.perf_counter() - thermal_start

        avg_temps.append(np.mean(T))
        exit_energies.append(KE_exit)

        # per-layer averages
        layer_temps = []
        for layer_i in range(NUM_PHYSICAL_LAYERS):
            start = layer_i * cells_per_layer
            end = start + cells_per_layer
            layer_temps.append(np.mean(T[start:end]))
        layer_avg_temps.append(layer_temps)

        if step_idx == 1 or step_idx % progress_interval == 0 or step_idx == num_steps:
            elapsed = time.perf_counter() - sim_start
            logger.info(
                "step %s/%s t=%.4f s elapsed=%.2f s transport=%.2f s thermal=%.2f s avg_temp=%.3f K exit_ke=%.2f MeV",
                step_idx,
                num_steps,
                t,
                elapsed,
                transport_total,
                thermal_total,
                avg_temps[-1],
                KE_exit,
            )

    logger.info(
        "simulation complete in %.2f s; transport %.2f s, thermal %.2f s",
        time.perf_counter() - sim_start,
        transport_total,
        thermal_total,
    )

    return times, np.array(avg_temps), np.array(exit_energies), np.array(layer_avg_temps), z, T


# ===================== RUN =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

times, avg_temp, exit_energy, layer_temps, z, T_final = run_simulation()

# --- Plots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].plot(times, avg_temp - 293, 'b-')
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("ΔT above ambient (K)")
axes[0, 0].set_title("Average Temperature Rise vs Time")

axes[0, 1].plot(times, exit_energy, 'r-')
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Exit KE (MeV)")
axes[0, 1].set_title("Transmitted Proton Kinetic Energy vs Time")

for i in range(NUM_PHYSICAL_LAYERS):
    axes[1, 0].plot(times, layer_temps[:, i] - 293, label=f"Layer {i+1}")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("ΔT above ambient (K)")
axes[1, 0].set_title("Per-Layer Temperature Rise")
axes[1, 0].legend()

axes[1, 1].plot(z * 1000, T_final - 293, 'k-', linewidth=0.5)
axes[1, 1].set_xlabel("Position (mm)")
axes[1, 1].set_ylabel("ΔT above ambient (K)")
axes[1, 1].set_title("Final Spatial Temperature Profile")
# mark layer boundaries
cells_per_layer = len(z) // NUM_PHYSICAL_LAYERS
for i in range(1, NUM_PHYSICAL_LAYERS):
    axes[1, 1].axvline(z[i * cells_per_layer] * 1000, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("simulation_results.png", dpi=150)
plt.show()

print(f"\nFinal avg temp: {avg_temp[-1]:.2f} K (ΔT = {avg_temp[-1]-293:.4f} K)")
print(f"Final exit energy: {exit_energy[-1]:.2f} MeV (of {PROTON_KE_MEV:.0f} MeV input)")
