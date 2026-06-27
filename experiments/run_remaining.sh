#!/bin/bash
# Runs the two remaining experiment groups in sequence:
#   1. mnist / no_entanglement  (~3.5h)
#   2. Patch QGAN: fashionmnist, mnist, breastmnist  (~3.5h total)

set -e
cd /home/mahlow/eq-gan
# OMP_NUM_THREADS=1 prevents nested parallelism with our Pool-16 workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== run_remaining started: $(date) ==="

# --- 1. mnist / no_entanglement ---
echo ""
echo ">>> [mnist / no_entanglement] started: $(date)"
/home/mahlow/anaconda3/envs/my_env/bin/python experiments/run_experiment.py \
  --dataset mnist --mode no_entanglement \
  --epochs 50 --batch_size 32 --n_train 5000 \
  --n_critic 3 --eval_every 10 --n_eval 1000 \
  --out_dir ./results --data_root ./data \
  2>&1 | tee results/logs/mnist_no_entanglement.log
echo ">>> [mnist / no_entanglement] done: $(date)"

# --- 2. Patch QGAN ---
run_patch() {
  local ds=$1
  echo ""
  echo ">>> [patchqgan / $ds] started: $(date)"
  /home/mahlow/anaconda3/envs/my_env/bin/python experiments/run_patch_qgan.py \
    --dataset "$ds" \
    --epochs 50 --batch_size 32 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 1000 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee "results/logs/patchqgan_${ds}.log"
  echo ">>> [patchqgan / $ds] done: $(date)"
}

run_patch fashionmnist
run_patch mnist
run_patch breastmnist

echo ""
echo "=== All remaining experiments done: $(date) ==="
