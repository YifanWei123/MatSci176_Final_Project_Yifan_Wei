#!/bin/bash
#SBATCH -J mace_train
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -o logs/mace_train_%j.out
#SBATCH -e logs/mace_train_%j.err

echo "==============================="
echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "==============================="

echo "SLURM job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"

echo ""
echo "===== CPU MEMORY STATUS ====="
free -h

echo ""
echo "===== GPU STATUS ====="
nvidia-smi

source ~/.bashrc
mamba activate mace_base

cd /home/groups/xlzheng/ywei2/MACE_training/final/simple_MD_epoch15/
mkdir -p logs

TRAIN=train_fps.extxyz
VALID=val_ood_r00.extxyz
TEST=test_ood_r00.extxyz
E0S='{8: 0.0, 20: 0.0, 26: 0.0, 27: 0.0, 38: 0.0, 39: 0.0, 56: 0.0, 57: 0.0, 60: 0.0, 62: 0.0, 64: 0.0}'

echo ""
echo "===== INPUT FILES ====="
ls -lh "$TRAIN" "$VALID" "$TEST"

echo ""
echo "===== START TRAINING ====="

mace_run_train \
  --name=full_model \
  --train_file="$TRAIN" \
  --valid_file="$VALID" \
  --energy_key=energy \
  --forces_key=forces \
  --E0s="$E0S" \
  --device=cuda \
  --max_num_epochs=16 \
  --batch_size=2 \
  --valid_batch_size=2

echo ""
echo "===== TRAINING FINISHED ====="
echo "End time: $(date)"

echo ""
echo "===== FINAL MEMORY STATUS ====="
free -h

echo ""
echo "===== FINAL GPU STATUS ====="
nvidia-smi



