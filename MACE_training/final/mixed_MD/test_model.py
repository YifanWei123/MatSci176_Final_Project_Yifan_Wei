import os
import glob
import time
import subprocess
from pathlib import Path

import ase.io
import numpy as np


# =========================
# User settings
# =========================
INPUT_FILE = "test_ood_r00.extxyz"
MODEL_FILE = "full_model.model"
CHUNK_DIR = "chunks"
PRED_DIR = "pred_chunks"
CHUNK_SIZE = 500
DEVICE = "cuda"
BATCH_SIZE = 8


# =========================
# Step 1: split input file
# =========================
def split_extxyz(input_file, chunk_dir, chunk_size):
    print("=" * 60)
    print("Step 1: Splitting input file into chunks")
    print("=" * 60)

    Path(chunk_dir).mkdir(exist_ok=True)

    atoms_list = ase.io.read(input_file, index=":")
    n_total = len(atoms_list)
    print(f"Loaded {n_total} frames from {input_file}")

    chunk_files = []
    for i in range(0, n_total, chunk_size):
        chunk_idx = i // chunk_size
        out_file = os.path.join(chunk_dir, f"test_ood_r00_chunk_{chunk_idx:02d}.extxyz")
        chunk_atoms = atoms_list[i:i + chunk_size]
        ase.io.write(out_file, chunk_atoms)
        chunk_files.append(out_file)
        print(f"Wrote {out_file} with {len(chunk_atoms)} frames")

    return chunk_files


# =========================
# Step 2: run mace_eval_configs
# =========================
def run_predictions(chunk_files, pred_dir, model_file, device="cuda", batch_size=8):
    print("\n" + "=" * 60)
    print("Step 2: Running predictions with mace_eval_configs")
    print("=" * 60)

    Path(pred_dir).mkdir(exist_ok=True)

    pred_files = []

    for chunk_file in chunk_files:
        base = Path(chunk_file).stem
        out_file = os.path.join(pred_dir, f"{base}_pred.extxyz")
        pred_files.append(out_file)

        print("========================================")
        print(f"Starting {base}")
        print(f"Input : {chunk_file}")
        print(f"Output: {out_file}")
        print(f"Time  : {time.ctime()}")

        start = time.time()

        cmd = [
            "mace_eval_configs",
            f"--configs={chunk_file}",
            f"--model={model_file}",
            f"--output={out_file}",
            f"--device={device}",
            f"--batch_size={batch_size}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        print(f"Finished command for {base}")
        print(f"Status: {result.returncode}")
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"Time: {time.ctime()}")

        if result.stdout:
            print("\n[STDOUT]")
            print(result.stdout)

        if result.stderr:
            print("\n[STDERR]")
            print(result.stderr)

        if result.returncode == 0:
            print(f"Done {base} in {elapsed:.1f}s")
        else:
            print(f"Failed {base} with status {result.returncode} after {elapsed:.1f}s")
            raise RuntimeError(f"Prediction failed for {chunk_file}")

    return pred_files


# =========================
# Step 3: evaluate metrics
# =========================
def evaluate_predictions(chunk_dir, pred_dir):
    print("\n" + "=" * 60)
    print("Step 3: Evaluating predictions")
    print("=" * 60)

    ref_files = sorted(glob.glob(os.path.join(chunk_dir, "test_ood_r00_chunk_*.extxyz")))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "test_ood_r00_chunk_*_pred.extxyz")))

    if len(ref_files) == 0:
        raise FileNotFoundError("No reference chunk files found.")
    if len(pred_files) == 0:
        raise FileNotFoundError("No prediction files found.")
    if len(ref_files) != len(pred_files):
        raise ValueError(f"Number of ref files ({len(ref_files)}) != pred files ({len(pred_files)})")

    dE = []
    dE_pa = []
    dF = []
    nframes = 0

    for ref, pred in zip(ref_files, pred_files):
        R = list(ase.io.iread(ref, ":"))
        P = list(ase.io.iread(pred, ":"))

        if len(R) != len(P):
            raise ValueError(f"Frame count mismatch: {ref}, {pred}, {len(R)}, {len(P)}")

        for r, p in zip(R, P):
            ref_energy = float(r.get_potential_energy())
            pred_energy = float(p.info["MACE_energy"])
            de = pred_energy - ref_energy

            dE.append(de)
            dE_pa.append(de / len(r))

            ref_forces = r.get_forces()
            pred_forces = p.arrays["MACE_forces"]
            dF.append(pred_forces - ref_forces)

            nframes += 1

    dE = np.array(dE)
    dE_pa = np.array(dE_pa)
    dF = np.concatenate(dF, axis=0)

    print("MY MODEL")
    print("frames:", nframes)
    print("Energy MAE          :", float(np.mean(np.abs(dE))), "eV")
    print("Energy RMSE         :", float(np.sqrt(np.mean(dE ** 2))), "eV")
    print("Per-atom Energy MAE :", float(np.mean(np.abs(dE_pa))), "eV/atom")
    print("Per-atom Energy RMSE:", float(np.sqrt(np.mean(dE_pa ** 2))), "eV/atom")
    print("Force MAE           :", float(np.mean(np.abs(dF))), "eV/Å")
    print("Force RMSE          :", float(np.sqrt(np.mean(dF ** 2))), "eV/Å")


# =========================
# Main
# =========================
if __name__ == "__main__":
    chunk_files = split_extxyz(INPUT_FILE, CHUNK_DIR, CHUNK_SIZE)
    run_predictions(
        chunk_files=chunk_files,
        pred_dir=PRED_DIR,
        model_file=MODEL_FILE,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )
    evaluate_predictions(CHUNK_DIR, PRED_DIR)
