import numpy as np
from ase.io import read, write
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.nptberendsen import NPTBerendsen
from mace.calculators import MACECalculator

print("=" * 80)
print("STEP 3: short NPT test")
print("=" * 80)

model_path = "full_model.model"
device = "cuda"

structure_file = "LSCF_Ov00_build.extxyz"

temperature_K = 300
pressure_bar = 1.01325
timestep_fs = 1.0
nsteps = 100
loginterval = 10

# compressibility:
# K ~ 100 GPa => beta = 1/K = 1e-11 Pa^-1
bulk_modulus_Pa = 100e9
compressibility_au = (1.0 / bulk_modulus_Pa) / units.Pascal

taut = 50.0 * units.fs
taup = 200.0 * units.fs

print(f"Model: {model_path}")
print(f"Structure: {structure_file}")
print(f"T = {temperature_K} K")
print(f"nsteps = {nsteps}")
print(f"compressibility_au = {compressibility_au}")

atoms = read(structure_file)
print("Loaded atoms:", len(atoms))
print("Initial cell lengths (A):", atoms.cell.lengths())
print("Initial volume (A^3):", atoms.get_volume())

calc = MACECalculator(
    model_paths=model_path,
    device=device,
    default_dtype="float32",
)
atoms.calc = calc

print("\nSanity before MD:")
e0 = atoms.get_potential_energy()
f0 = atoms.get_forces()
s0 = atoms.get_stress(voigt=True)

print("Initial energy (eV):", e0)
print("Initial max |F| (eV/A):", np.linalg.norm(f0, axis=1).max())
print("Initial stress (Voigt, eV/A^3):", s0)

print("\nInitializing velocities...")
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
Stationary(atoms)
ZeroRotation(atoms)

traj_file = "LSCF_300K_short_npt.traj"
log_file = "LSCF_300K_short_npt.log"

dyn = NPTBerendsen(
    atoms,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature_K,
    pressure_au=pressure_bar * units.bar,
    taut=taut,
    taup=taup,
    compressibility_au=compressibility_au,
    trajectory=traj_file,
    logfile=log_file,
    loginterval=loginterval,
)

def print_status():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * len(atoms) * units.kB)
    vol = atoms.get_volume()
    cell = atoms.cell.lengths()
    print(
        f"STEP status | "
        f"Epot={epot:.6f} eV | "
        f"Ekin={ekin:.6f} eV | "
        f"Tinst={temp:.2f} K | "
        f"V={vol:.3f} A^3 | "
        f"a,b,c={cell}"
    )

dyn.attach(print_status, interval=10)

print("\nRunning short NPT...")
dyn.run(nsteps)

print("\nFinished short NPT.")
print("Final cell lengths (A):", atoms.cell.lengths())
print("Final volume (A^3):", atoms.get_volume())

write("LSCF_300K_short_npt_final.extxyz", atoms)
print("Wrote: LSCF_300K_short_npt_final.extxyz")
print("Wrote:", traj_file)
print("Wrote:", log_file)
