from ase.io import read
from mace.calculators import MACECalculator
import numpy as np

print("=" * 80)
print("STEP 2: MACE sanity check")
print("=" * 80)

model_path = "full_model.model"
device = "cuda"   # 没 GPU 就改成 "cpu"

structure_files = [
    "LSCF_Ov00_build.extxyz",
    "CaNdY_Ov00_build.extxyz",
]

print(f"Using model: {model_path}")
print(f"Using device: {device}")

calc = MACECalculator(
    model_paths=model_path,
    device=device,
    default_dtype="float32",
)

for f in structure_files:
    print("\n" + "=" * 80)
    print(f"Reading structure: {f}")
    print("=" * 80)

    atoms = read(f)
    atoms.calc = calc

    print("Number of atoms:", len(atoms))
    print("Cell lengths (A):", atoms.cell.lengths())
    print("Volume (A^3):", atoms.get_volume())

    print("\nComputing potential energy...")
    e = atoms.get_potential_energy()
    print("Potential energy (eV):", e)
    print("Per-atom energy (eV/atom):", e / len(atoms))

    print("\nComputing forces...")
    F = atoms.get_forces()
    print("Forces shape:", F.shape)
    print("Force norm stats (eV/A):")
    fnorm = np.linalg.norm(F, axis=1)
    print("  min :", fnorm.min())
    print("  max :", fnorm.max())
    print("  mean:", fnorm.mean())

    print("\nComputing stress...")
    try:
        s = atoms.get_stress(voigt=True)
        print("Stress (Voigt, eV/A^3):", s)
    except Exception as err:
        print("FAILED to get stress.")
        print("Error:", repr(err))

print("\nDone with STEP 2.")
