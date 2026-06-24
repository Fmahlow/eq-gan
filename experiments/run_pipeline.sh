#!/bin/bash
# Sequential experiment pipeline.
# Waits for the EQ-GAN full (FashionMNIST) run to finish,
# then runs ablations and remaining datasets in order.

set -e
cd /workspace/quantum-gan-paper

LOGDIR="results/logs"
FULL_LOG="$LOGDIR/fashionmnist_full.log"

echo "=== Pipeline started: $(date) ==="
echo "Waiting for EQ-GAN full (FashionMNIST) to complete..."

# Wait until the full run log contains "Done." (written at the end of run_experiment.py)
until grep -q "^Done\." "$FULL_LOG" 2>/dev/null; do
    sleep 60
done
echo "EQ-GAN full (FashionMNIST) done: $(date)"

# ---- Ablation 1: no_entanglement (FashionMNIST) ----
echo ""
echo ">>> Running ablation: no_entanglement / FashionMNIST [$(date)]"
python3 experiments/run_experiment.py \
    --dataset fashionmnist --mode no_entanglement \
    --epochs 50 --batch_size 32 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 1000 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee "$LOGDIR/fashionmnist_no_entanglement.log"
echo ">>> no_entanglement done: $(date)"

# ---- Ablation 2: no_style (FashionMNIST) ----
echo ""
echo ">>> Running ablation: no_style / FashionMNIST [$(date)]"
python3 experiments/run_experiment.py \
    --dataset fashionmnist --mode no_style \
    --epochs 50 --batch_size 32 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 1000 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee "$LOGDIR/fashionmnist_no_style.log"
echo ">>> no_style done: $(date)"

# ---- EQ-GAN full: MNIST ----
echo ""
echo ">>> Running EQ-GAN full / MNIST [$(date)]"
python3 experiments/run_experiment.py \
    --dataset mnist --mode full \
    --epochs 50 --batch_size 32 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 1000 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee "$LOGDIR/mnist_full.log"
echo ">>> MNIST full done: $(date)"

# ---- EQ-GAN no_entanglement: MNIST ----
echo ""
echo ">>> Running ablation: no_entanglement / MNIST [$(date)]"
python3 experiments/run_experiment.py \
    --dataset mnist --mode no_entanglement \
    --epochs 50 --batch_size 32 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 1000 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee "$LOGDIR/mnist_no_entanglement.log"
echo ">>> MNIST no_entanglement done: $(date)"

# ---- EQ-GAN full: BreastMNIST ----
echo ""
echo ">>> Running EQ-GAN full / BreastMNIST [$(date)]"
python3 experiments/run_experiment.py \
    --dataset breastmnist --mode full \
    --epochs 50 --batch_size 32 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 500 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee "$LOGDIR/breastmnist_full.log"
echo ">>> BreastMNIST full done: $(date)"

# ---- Classical baselines: MNIST and BreastMNIST ----
echo ""
echo ">>> Running WGAN-GP / MNIST [$(date)]"
python3 experiments/run_classical_baseline.py \
    --dataset mnist --model wgan_gp \
    --epochs 50 --batch_size 64 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 1000 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee "$LOGDIR/mnist_wgan_gp.log"

echo ""
echo ">>> Running DCGAN / MNIST [$(date)]"
python3 experiments/run_classical_baseline.py \
    --dataset mnist --model dcgan \
    --epochs 50 --batch_size 64 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 1000 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee "$LOGDIR/mnist_dcgan.log"

echo ""
echo ">>> Running WGAN-GP / BreastMNIST [$(date)]"
python3 experiments/run_classical_baseline.py \
    --dataset breastmnist --model wgan_gp \
    --epochs 50 --batch_size 64 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 500 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee "$LOGDIR/breastmnist_wgan_gp.log"

echo ""
echo "=== Pipeline completed: $(date) ==="
echo "All results in: /workspace/quantum-gan-paper/results/"
