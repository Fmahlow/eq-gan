#!/bin/bash
# Patch QGAN sequential pipeline — runs after quantum pipeline finishes.
# Waits for mnist_no_entanglement to complete, then runs 3 Patch QGAN experiments.

set -e
cd /home/mahlow/eq-gan
export OMP_NUM_THREADS=20

echo "=== Patch QGAN pipeline started: $(date) ==="

# Wait for the quantum sequential pipeline to finish
echo "Waiting for mnist_no_entanglement to complete..."
while ps aux | grep "run_sequential\|run_experiment" | grep -v grep | grep -q "python3"; do
  sleep 60
done
echo "Quantum pipeline done. Starting Patch QGAN: $(date)"

run_patch() {
  local ds=$1
  echo ""
  echo ">>> [patchqgan / $ds] started: $(date)"
  python3 experiments/run_patch_qgan.py \
    --dataset $ds \
    --epochs 50 --batch_size 32 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 1000 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee results/logs/patchqgan_${ds}.log
  echo ">>> [patchqgan / $ds] done: $(date)"
}

run_patch fashionmnist
run_patch mnist
run_patch breastmnist

echo ""
echo "=== Patch QGAN pipeline done: $(date) ==="
