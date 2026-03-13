import os
import json
import numpy as np
import pandas as pd

from ase import Atoms, units
from ase.io import write, read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.nptberendsen import NPTBerendsen

# ======================================================================
# PRETRAINED MACE CALCULATOR
# ======================================================================
# Option A: most common pretrained MACE-MP usage
from mace.calculators import mace_mp

print("=" * 100)
print("PRETRAINED MACE MD FOR SELECTED CONDITIONS")
print("=" * 100)

# ======================================================================================
# USER SETTINGS
# ======================================================================================
device = "cuda"   # change to "cpu" if needed

a0 = 3.90
supercell = (5, 5, 5)   # 125 formula units
seed_base = 1000

# Materials
configurations = {
    "LSCF": {
        "A_species": ["La", "Sr"],
        "A_frac":    [0.6, 0.4],
        "B_species": ["Co", "Fe"],
        "B_frac":    [0.2, 0.8],
    },
    "A5mix": {
        "A_species": ["Ca", "Nd", "Y", "La", "Sr"],
        "A_frac":    [0.2, 0.2, 0.2, 0.2, 0.2],
        "B_species": ["Co", "Fe"],
        "B_frac":    [0.2, 0.8],
    },
}

# Only selected temperature-Ov combinations
selected_conditions = {
    "LSCF": [
        (26, 0),
        (100, 0),
        (200, 0),
        (300, 0),
        (400, 0),
        (500, 0),
        (600, 1),
        (700, 2),
        (800, 3),
    ],
    "A5mix": [
        (26, 0),
        (100, 0),
        (200, 0),
        (300, 0),
        (400, 0),
        (500, 1),
        (600, 4),
        (700, 7),
        (800, 10),
    ],
}

# NPT settings: same as previous
pressure_bar = 1.01325
md_timestep_fs = 1.0
nsteps_total = 500
loginterval = 20
traj_interval = 20
taut_fs = 10.0
taup_fs = 100.0

# compressibility estimate
bulk_modulus_Pa = 100e9
compressibility_au = (1.0 / bulk_modulus_Pa) / units.Pascal

outdir = "pretrained_mace_selected_conditions"
traj_dir = os.path.join(outdir, "traj")
xyz_dir = os.path.join(outdir, "xyz")
table_dir = os.path.join(outdir, "tables")

os.makedirs(outdir, exist_ok=True)
os.makedirs(traj_dir, exist_ok=True)
os.makedirs(xyz_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

print(f"Device: {device}")
print(f"Supercell: {supercell}")
print(f"MD timestep = {md_timestep_fs} fs")
print(f"Total steps = {nsteps_total}")
print(f"traj interval = {traj_interval}")
print(f"log interval = {loginterval}")
print(f"taut = {taut_fs} fs")
print(f"taup = {taup_fs} fs")
print(f"compressibility_au = {compressibility_au}")
print()

# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================
def build_cubic_perovskite_ab_o3_tagged(a=3.90, A="La", B="Co"):
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
    atoms.set_tags([1, 2, 3, 3, 3])  # A=1, B=2, O=3
    return atoms

def indices_by_tag(atoms, tag_value):
    return [i for i, t in enumerate(atoms.get_tags()) if t == tag_value]

def assign_species_by_counts(atoms, site_indices, species, fractions, seed=0):
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

def build_material_structure(cfg, base_atoms, a_idx, b_idx, o_idx, n_vac, seed_offset=0):
    atoms = base_atoms.copy()
    n_fu = len(a_idx)

    seed_A = seed_base + seed_offset + 1
    seed_B = seed_base + seed_offset + 2
    seed_O = seed_base + seed_offset + 3

    a_counts = assign_species_by_counts(
        atoms, a_idx,
        species=cfg["A_species"],
        fractions=cfg["A_frac"],
        seed=seed_A
    )

    b_counts = assign_species_by_counts(
        atoms, b_idx,
        species=cfg["B_species"],
        fractions=cfg["B_frac"],
        seed=seed_B
    )

    n_vac_real, delta_eff, removed = apply_oxygen_vacancies_by_count(
        atoms, o_idx, n_vac=n_vac, n_fu=n_fu, seed=seed_O
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
print()

# ======================================================================================
# INITIALIZE PRETRAINED MACE
# ======================================================================================
print("=" * 100)
print("INITIALIZING PRETRAINED MACE")
print("=" * 100)

calc = mace_mp(device=device)

print("Pretrained MACE calculator ready.")
print()

# ======================================================================================
# MAIN LOOP
# ======================================================================================
all_records = []

for i_mat, (mat_name, cfg) in enumerate(configurations.items()):
    print("\n" + "=" * 100)
    print(f"START MATERIAL: {mat_name}")
    print("=" * 100)

    records_this_material = []

    for icond, (T_C, n_vac) in enumerate(selected_conditions[mat_name]):
        T_K = T_C + 273.15
        tag = condition_tag(mat_name, T_C, n_vac)
        seed_offset = 10000 * i_mat + 100 * icond

        print("\n" + "-" * 100)
        print(f"Running condition: {tag}")
        print(f"T = {T_C} C  ({T_K:.2f} K), Ov = {n_vac}")
        print("-" * 100)

        # ----------------------------------------------------------
        # Build structure
        # ----------------------------------------------------------
        atoms, a_counts, b_counts, n_vac_real, delta_eff, removed = build_material_structure(
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
        print(f"  Ov input = {n_vac}")
        print(f"  Ov realized = {n_vac_real}")
        print(f"  delta_eff = {delta_eff:.6f}")

        print("  A-site counts:")
        for sp, c in zip(cfg["A_species"], a_counts):
            print(f"    {sp}: {int(c)}")

        print("  B-site counts:")
        for sp, c in zip(cfg["B_species"], b_counts):
            print(f"    {sp}: {int(c)}")

        # ----------------------------------------------------------
        # Attach calculator and sanity check
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # Initialize velocities
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # Analyze trajectory
        # ----------------------------------------------------------
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

        write(xyz_path, frames)
        print(f"  Wrote traj: {traj_path}")
        print(f"  Wrote xyz : {xyz_path}")
        print(f"  Wrote log : {log_path}")

        rec = {
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
        }

        records_this_material.append(rec)
        all_records.append(rec)

    # save per-material table
    df_mat = pd.DataFrame(records_this_material)
    mat_csv = os.path.join(table_dir, f"{mat_name}_selected_conditions_table.csv")
    df_mat.to_csv(mat_csv, index=False)
    print(f"\nWrote material table: {mat_csv}")

# save combined table
df_all = pd.DataFrame(all_records)
all_csv = os.path.join(table_dir, "all_selected_conditions_table.csv")
all_json = os.path.join(table_dir, "all_selected_conditions_table.json")

df_all.to_csv(all_csv, index=False)
with open(all_json, "w") as f:
    json.dump(all_records, f, indent=2)

print("\n" + "=" * 100)
print("ALL DONE")
print("=" * 100)
print(f"Wrote combined table: {all_csv}")
print(f"Wrote combined json : {all_json}")
