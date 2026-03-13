import os
import json
import numpy as np
import pandas as pd
from collections import Counter

from ase import Atoms, units
from ase.io import write, read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.nptberendsen import NPTBerendsen

from mace.calculators import MACECalculator

print("=" * 100)
print("STEP 4: T(C) x Ov MATRIX -> equilibrium volume")
print("=" * 100)

# ======================================================================================
# USER SETTINGS
# ======================================================================================
model_path = "full_model.model"
device = "cuda"   # change to "cpu" if needed

a0 = 3.90
supercell = (5, 5, 5)   # 125 formula units
seed_base = 1000

# Two materials only
configurations = {
    "LSCF": {
        "A_species": ["La", "Sr"],
        "A_frac":    [0.6, 0.4],
        "B_species": ["Co", "Fe"],
        "B_frac":    [0.2, 0.8],
    },
    "CaNdY": {
        "A_species": ["Ca", "Nd", "Y" , "La", "Sr"],
        "A_frac":    [0.2 , 0.2 , 0.2 , 0.2 , 0.2],
        "B_species": ["Co", "Fe"],
        "B_frac":    [0.2, 0.8],
    },
}

# Temperature axis in Celsius
temperatures_C = [26] + list(range(100, 801, 100))
# Used for MD
temperatures_K = [t + 273.15 for t in temperatures_C]

# Same Ov list for both materials
n_vac_list = [0, 1, 2, 3, 4, 7, 10]

# NPT settings
pressure_bar = 1.01325

md_timestep_fs = 1.0     # actual MD timestep
nsteps_total = 500       # total number of MD steps
loginterval = 20         # print to log every 20 steps
traj_interval = 20       # save one frame every 20 steps

taut_fs = 10.0
taup_fs = 100.0

# Compressibility estimate
bulk_modulus_Pa = 100e9
compressibility_au = (1.0 / bulk_modulus_Pa) / units.Pascal

outdir = "mace_T_Ov_matrix"
traj_dir = os.path.join(outdir, "traj")
xyz_dir = os.path.join(outdir, "xyz")
table_dir = os.path.join(outdir, "tables")

os.makedirs(outdir, exist_ok=True)
os.makedirs(traj_dir, exist_ok=True)
os.makedirs(xyz_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

print(f"Model path: {model_path}")
print(f"Device: {device}")
print(f"Supercell: {supercell}")
print(f"Temperatures (C): {temperatures_C}")
print(f"Temperatures (K): {[round(x, 2) for x in temperatures_K]}")
print(f"O vacancy list: {n_vac_list}")
print(f"MD timestep = {md_timestep_fs} fs")
print(f"Total steps = {nsteps_total}")
print(f"Trajectory save interval = every {traj_interval} steps")
print(f"Log interval = every {loginterval} steps")
print(f"taut = {taut_fs} fs")
print(f"taup = {taup_fs} fs")
print(f"compressibility_au = {compressibility_au}")
print()

# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================
def build_cubic_perovskite_ab_o3_tagged(a=3.90, A="La", B="Co"):
    """
    Primitive cubic ABO3 with tags:
      A-site tag = 1
      B-site tag = 2
      O-site tag = 3
    """
    symbols = [A, B, "O", "O", "O"]
    scaled_positions = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]
    cell = np.eye(3) * a
    atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)
    atoms.set_tags([1, 2, 3, 3, 3])
    return atoms

def indices_by_tag(atoms, tag_value):
    return [i for i, t in enumerate(atoms.get_tags()) if t == tag_value]

def assign_species_by_counts(atoms, site_indices, species, fractions, seed=0):
    """
    Convert fractions to exact integer counts, then randomly assign.
    """
    rng = np.random.default_rng(seed)
    n = len(site_indices)

    fractions = np.array(fractions, dtype=float)
    fractions = fractions / fractions.sum()

    raw = fractions * n
    counts = np.floor(raw).astype(int)
    remain = n - counts.sum()

    if remain > 0:
        frac_part = raw - np.floor(raw)
        order = np.argsort(frac_part)[::-1]
        for i in order[:remain]:
            counts[i] += 1

    assert counts.sum() == n, f"Count mismatch: {counts.sum()} != {n}"

    assignment = []
    for sp, c in zip(species, counts):
        assignment += [sp] * int(c)

    assignment = np.array(assignment, dtype=object)
    rng.shuffle(assignment)

    for idx, sp in zip(site_indices, assignment):
        atoms[idx].symbol = sp

    return counts

def apply_oxygen_vacancies_by_count(atoms, o_idx, n_vac, n_fu, seed=0):
    """
    Remove an integer number of oxygen vacancies directly.
    For ABO_(3-delta), delta_eff = n_vac / n_fu
    """
    rng = np.random.default_rng(seed)

    if n_vac == 0:
        return 0, 0.0, []

    if n_vac > len(o_idx):
        raise ValueError(f"n_vac={n_vac} > number of O sites={len(o_idx)}")

    remove = rng.choice(o_idx, size=n_vac, replace=False).tolist()
    for idx in sorted(remove, reverse=True):
        del atoms[idx]

    delta_eff = n_vac / n_fu
    return n_vac, delta_eff, sorted(remove)

def build_material_structure(mat_name, cfg, base_atoms, a_idx, b_idx, o_idx, n_vac, seed_offset=0):
    atoms = base_atoms.copy()
    n_fu = len(a_idx)

    seed_A = seed_base + seed_offset + 1
    seed_B = seed_base + seed_offset + 2
    seed_O = seed_base + seed_offset + 3

    a_counts = assign_species_by_counts(
        atoms,
        a_idx,
        species=cfg["A_species"],
        fractions=cfg["A_frac"],
        seed=seed_A
    )

    b_counts = assign_species_by_counts(
        atoms,
        b_idx,
        species=cfg["B_species"],
        fractions=cfg["B_frac"],
        seed=seed_B
    )

    n_vac_real, delta_eff, removed = apply_oxygen_vacancies_by_count(
        atoms,
        o_idx,
        n_vac=n_vac,
        n_fu=n_fu,
        seed=seed_O
    )

    return atoms, a_counts, b_counts, n_vac_real, delta_eff, removed

def condition_tag(mat_name, T_C, n_vac):
    return f"{mat_name}__T{int(round(T_C)):03d}C__Ov{n_vac:02d}"

# ======================================================================================
# BUILD BASE SUPERCELL
# ======================================================================================
prim = build_cubic_perovskite_ab_o3_tagged(a=a0, A="La", B="Co")
base_atoms = prim.repeat(supercell)

a_idx = indices_by_tag(base_atoms, 1)
b_idx = indices_by_tag(base_atoms, 2)
o_idx = indices_by_tag(base_atoms, 3)

n_fu = len(a_idx)

print("=" * 100)
print("BASE SUPERCELL INFO")
print("=" * 100)
print(f"Total atoms before doping/vacancy = {len(base_atoms)}")
print(f"A sites = {len(a_idx)}")
print(f"B sites = {len(b_idx)}")
print(f"O sites = {len(o_idx)}")
print(f"Formula units = {n_fu}")
print("Ov to delta_eff mapping for current supercell:")
for nv in n_vac_list:
    print(f"  Ov = {nv:2d}  -> delta_eff = {nv / n_fu:.5f}")
print()

# ======================================================================================
# INITIALIZE CALCULATOR
# ======================================================================================
print("=" * 100)
print("INITIALIZING MACE CALCULATOR")
print("=" * 100)

calc = MACECalculator(
    model_paths=model_path,
    device=device,
    default_dtype="float32",
)

print("Calculator ready.")
print()

# ======================================================================================
# MAIN LOOP
# ======================================================================================
for i_mat, (mat_name, cfg) in enumerate(configurations.items()):
    print("\n" + "=" * 100)
    print(f"START MATERIAL: {mat_name}")
    print("=" * 100)

    volume_matrix = pd.DataFrame(index=temperatures_C, columns=n_vac_list, dtype=float)
    delta_eff_matrix = pd.DataFrame(index=temperatures_C, columns=n_vac_list, dtype=float)

    records = []

    for iT, T_C in enumerate(temperatures_C):
        T_K = T_C + 273.15

        print("\n" + "-" * 100)
        print(f"T = {T_C} C   ({T_K:.2f} K)")
        print("-" * 100)

        for ivac, n_vac in enumerate(n_vac_list):
            tag = condition_tag(mat_name, T_C, n_vac)
            seed_offset = 10000 * i_mat + 1000 * iT + 100 * ivac

            print(f"\nRunning condition: {tag}")

            # ------------------------------------------------------------------
            # Build structure
            # ------------------------------------------------------------------
            atoms, a_counts, b_counts, n_vac_real, delta_eff, removed = build_material_structure(
                mat_name=mat_name,
                cfg=cfg,
                base_atoms=base_atoms,
                a_idx=a_idx,
                b_idx=b_idx,
                o_idx=o_idx,
                n_vac=n_vac,
                seed_offset=seed_offset
            )

            print(f"  Built atoms: N = {len(atoms)}")
            print(f"  Cell lengths (A): {atoms.cell.lengths()}")
            print(f"  Volume before MD (A^3): {atoms.get_volume():.6f}")
            print(f"  Input Ov = {n_vac}")
            print(f"  Realized Ov = {n_vac_real}")
            print(f"  delta_eff = {delta_eff:.6f}")

            print("  A-site counts:")
            for sp, c in zip(cfg["A_species"], a_counts):
                print(f"    {sp}: {int(c)}")

            print("  B-site counts:")
            for sp, c in zip(cfg["B_species"], b_counts):
                print(f"    {sp}: {int(c)}")

            # ------------------------------------------------------------------
            # Attach calculator and do sanity check
            # ------------------------------------------------------------------
            atoms.calc = calc

            print("  Sanity check before MD...")
            e0 = atoms.get_potential_energy()
            f0 = atoms.get_forces()
            fnorm = np.linalg.norm(f0, axis=1)

            try:
                s0 = atoms.get_stress(voigt=True)
            except Exception as err:
                print("  FAILED to get stress before MD.")
                print("  Error:", repr(err))
                raise

            print(f"  Initial energy (eV): {e0:.6f}")
            print(f"  Initial per-atom energy (eV/atom): {e0 / len(atoms):.6f}")
            print(f"  Initial |F| min/max/mean (eV/A): {fnorm.min():.6f} / {fnorm.max():.6f} / {fnorm.mean():.6f}")
            print(f"  Initial stress (Voigt, eV/A^3): {s0}")

            # ------------------------------------------------------------------
            # Initialize velocities
            # ------------------------------------------------------------------
            MaxwellBoltzmannDistribution(atoms, temperature_K=T_K)
            Stationary(atoms)
            ZeroRotation(atoms)

            traj_path = os.path.join(traj_dir, f"{tag}.traj")
            xyz_path = os.path.join(xyz_dir, f"{tag}.extxyz")
            log_path = os.path.join(traj_dir, f"{tag}.log")

            dyn = NPTBerendsen(
                atoms,
                timestep=md_timestep_fs * units.fs,
                temperature_K=T_K,
                pressure_au=pressure_bar * units.bar,
                taut=taut_fs * units.fs,
                taup=taup_fs * units.fs,
                compressibility_au=compressibility_au,
                trajectory=traj_path,
                logfile=log_path,
                loginterval=loginterval,
            )

            def print_status():
                epot = atoms.get_potential_energy()
                ekin = atoms.get_kinetic_energy()
                temp_inst = ekin / (1.5 * len(atoms) * units.kB)
                vol = atoms.get_volume()
                abc = atoms.cell.lengths()
                print(
                    f"    STATUS | "
                    f"Epot={epot:.6f} eV | "
                    f"Ekin={ekin:.6f} eV | "
                    f"Tinst={temp_inst:.2f} K | "
                    f"V={vol:.6f} A^3 | "
                    f"a,b,c={abc}"
                )

            dyn.attach(print_status, interval=traj_interval)

            print(f"  Starting NPT: total steps = {nsteps_total}")
            dyn.run(nsteps_total)
            print("  NPT finished.")

            # ------------------------------------------------------------------
            # Analyze trajectory
            # ------------------------------------------------------------------
            frames = read(traj_path, index=":")
            print(f"  Number of saved frames in traj = {len(frames)}")

            volumes = np.array([a.get_volume() for a in frames], dtype=float)
            cell_lengths = np.array([a.cell.lengths() for a in frames], dtype=float)

            V_mean = float(volumes.mean())
            V_std = float(volumes.std())
            abc_mean = cell_lengths.mean(axis=0)
            abc_std = cell_lengths.std(axis=0)

            print(f"  Equilibrium mean volume (A^3) = {V_mean:.6f}")
            print(f"  Equilibrium std  volume (A^3) = {V_std:.6f}")
            print(f"  Mean a,b,c (A) = {abc_mean}")
            print(f"  Std  a,b,c (A) = {abc_std}")

            # Save all saved frames as extxyz
            write(xyz_path, frames)
            print(f"  Wrote traj: {traj_path}")
            print(f"  Wrote xyz : {xyz_path}")
            print(f"  Wrote log : {log_path}")

            # Fill matrices
            volume_matrix.loc[T_C, n_vac] = V_mean
            delta_eff_matrix.loc[T_C, n_vac] = delta_eff

            # Detailed record
            records.append({
                "material": mat_name,
                "T_C": T_C,
                "T_K": T_K,
                "n_vac_input": n_vac,
                "n_vac_real": n_vac_real,
                "delta_eff": delta_eff,
                "n_atoms": len(atoms),
                "V_mean_A3": V_mean,
                "V_std_A3": V_std,
                "a_mean_A": float(abc_mean[0]),
                "b_mean_A": float(abc_mean[1]),
                "c_mean_A": float(abc_mean[2]),
                "a_std_A": float(abc_std[0]),
                "b_std_A": float(abc_std[1]),
                "c_std_A": float(abc_std[2]),
                "traj_path": traj_path,
                "xyz_path": xyz_path,
                "log_path": log_path,
            })

    # ==================================================================================
    # SAVE TABLES FOR THIS MATERIAL
    # ==================================================================================
    print("\n" + "=" * 100)
    print(f"SAVING TABLES FOR {mat_name}")
    print("=" * 100)

    volume_matrix.index.name = "T_C"
    delta_eff_matrix.index.name = "T_C"

    volume_csv = os.path.join(table_dir, f"{mat_name}_volume_matrix.csv")
    delta_eff_csv = os.path.join(table_dir, f"{mat_name}_delta_eff_matrix.csv")
    long_csv = os.path.join(table_dir, f"{mat_name}_long_table.csv")
    json_path = os.path.join(table_dir, f"{mat_name}_records.json")

    volume_matrix.to_csv(volume_csv)
    delta_eff_matrix.to_csv(delta_eff_csv)

    df_long = pd.DataFrame(records)
    df_long.to_csv(long_csv, index=False)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote volume matrix   : {volume_csv}")
    print(f"Wrote delta_eff matrix: {delta_eff_csv}")
    print(f"Wrote long table      : {long_csv}")
    print(f"Wrote json records    : {json_path}")

print("\n" + "=" * 100)
print("ALL DONE")
print("=" * 100)
