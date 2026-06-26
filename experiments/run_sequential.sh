#!/bin/bash
# Sequential quantum experiment pipeline — single process with 20 threads.
# ~2.5h per experiment, ~12h total for 5 runs.

set -e
cd /home/mahlow/eq-gan
export OMP_NUM_THREADS=20

echo "=== Sequential pipeline started: $(date) ==="

run_exp() {
  local ds=$1 mode=$2 n_eval=${3:-1000}
  echo ""
  echo ">>> [$ds / $mode] started: $(date)"
  /home/mahlow/anaconda3/envs/my_env/bin/python experiments/run_experiment.py \
    --dataset $ds --mode $mode \
    --epochs 50 --batch_size 32 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval $n_eval \
    --out_dir ./results --data_root ./data \
    2>&1 | tee results/logs/${ds}_${mode}.log
  echo ">>> [$ds / $mode] done: $(date)"
}

run_exp fashionmnist full        1000
run_exp fashionmnist no_entanglement 1000
run_exp fashionmnist no_style    1000
run_exp mnist       full        1000
run_exp mnist       no_entanglement 1000

echo ""
echo "=== All done: $(date) ==="
