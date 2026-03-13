import numpy as np
from collections import Counter
from ase import Atoms
from ase.io import write

print("=" * 80)
print("STEP 1: Build tagged cubic perovskite supercells")
print("=" * 80)

# ------------------------------------------------------------------
# user settings
# ------------------------------------------------------------------
a0 = 3.90
supercell = (5, 5, 5)
seed_base = 1000

configurations = {
    "LSCF": {
        "A_species": ["La", "Sr"],
        "A_frac":    [0.6, 0.4],
        "B_species": ["Co", "Fe"],
        "B_frac":    [0.2, 0.8],
    },
    "CaNdY": {
        "A_species": ["Ca", "Nd", "Y" , "La", "Sr"],
        "A_frac":    [0.2,0.2,0.2,0.2,0.2],
        "B_species": ["Co", "Fe"],
        "B_frac":    [0.2, 0.8],
    },
}


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
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
    Convert fractions to exact integer counts and assign randomly.
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

def apply_oxygen_vacancies(atoms, o_idx, n_vac=0, seed=0):
    if n_vac == 0:
        return []

    rng = np.random.default_rng(seed)
    remove = rng.choice(o_idx, size=n_vac, replace=False).tolist()

    for idx in sorted(remove, reverse=True):
        del atoms[idx]

    return remove

def quick_check(name, atoms, removed):
    counts = Counter(atoms.get_chemical_symbols())
    print("-" * 80)
    print(f"Material: {name}")
    print(f"Total atoms: {len(atoms)}")
    print(f"Cell lengths (A): {atoms.cell.lengths()}")
    print(f"Cell angles  (deg): {atoms.cell.angles()}")
    print(f"Removed O count: {len(removed)}")
    print("Element counts:", dict(counts))
    print("Volume (A^3):", atoms.get_volume())

# ------------------------------------------------------------------
# build base supercell
# ------------------------------------------------------------------
prim = build_cubic_perovskite_ab_o3_tagged(a=a0, A="La", B="Co")
base_atoms = prim.repeat(supercell)

a_idx = indices_by_tag(base_atoms, 1)
b_idx = indices_by_tag(base_atoms, 2)
o_idx = indices_by_tag(base_atoms, 3)

print("Base supercell built.")
print("Supercell =", supercell)
print("Total atoms before doping/vacancy =", len(base_atoms))
print("A sites =", len(a_idx))
print("B sites =", len(b_idx))
print("O sites =", len(o_idx))
print("Expected total = A + B + O =", len(a_idx) + len(b_idx) + len(o_idx))

# ------------------------------------------------------------------
# generate structures
# ------------------------------------------------------------------
for i_mat, (name, cfg) in enumerate(configurations.items()):
    print("\n" + "=" * 80)
    print(f"Generating structure for {name}")
    print("=" * 80)

    atoms = base_atoms.copy()

    seed_A = seed_base + 100 * i_mat + 1
    seed_B = seed_base + 100 * i_mat + 2
    seed_O = seed_base + 100 * i_mat + 3

    a_counts = assign_species_by_counts(
        atoms,
        a_idx,
        species=cfg["A_species"],
        fractions=cfg["A_frac"],
        seed=seed_A,
    )
    b_counts = assign_species_by_counts(
        atoms,
        b_idx,
        species=cfg["B_species"],
        fractions=cfg["B_frac"],
        seed=seed_B,
    )

    removed = apply_oxygen_vacancies(atoms, o_idx, n_vac=0, seed=seed_O)

    print("A-site assignment:")
    for sp, c in zip(cfg["A_species"], a_counts):
        print(f"  {sp}: {int(c)}")

    print("B-site assignment:")
    for sp, c in zip(cfg["B_species"], b_counts):
        print(f"  {sp}: {int(c)}")

    quick_check(name, atoms, removed)

    out_xyz = f"{name}_Ov00_build.extxyz"
    out_cif = f"{name}_Ov00_build.cif"

    write(out_xyz, atoms)
    write(out_cif, atoms)

    print(f"Wrote: {out_xyz}")
    print(f"Wrote: {out_cif}")

print("\nDone with STEP 1.")

