#!/bin/bash
set -e
cd /home/mahlow/eq-gan
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== run_patchqgan_only started: $(date) ==="

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
echo "=== All Patch QGAN done: $(date) ==="
