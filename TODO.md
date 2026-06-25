# TODO — EQ-GAN Paper: Pending Experiments & Next Steps

## Context

This is a quantum GAN paper investigating whether quantum entanglement improves
conditional image generation. The main finding (already in the paper) is a
**negative result**: the no-entanglement variant consistently beats the full
EQ-GAN (FID 173.7 vs 192.6 on FashionMNIST), and circuit analysis explains why.

The paper draft is at `paper/src/main.tex` and is substantially complete.
All results so far are in `results/checkpoints/*/metrics_epoch*.json`.

---

## Experiments Left to Run

### 1. `mnist / no_entanglement` — INTERRUPTED at epoch 5/50

Run sequentially with 20 CPU threads:

```bash
cd /workspace/quantum-gan-paper
export OMP_NUM_THREADS=20
python3 experiments/run_experiment.py \
  --dataset mnist --mode no_entanglement \
  --epochs 50 --batch_size 32 --n_train 5000 \
  --n_critic 3 --eval_every 10 --n_eval 1000 \
  --out_dir ./results --data_root ./data \
  2>&1 | tee results/logs/mnist_no_entanglement.log
```

Expected: ~3.5h (50 epochs × ~190s). Final metric goes into Table 1.

---

### 2. Patch QGAN — 3 datasets, NOT YET STARTED

Run sequentially after mnist/no_entanglement:

```bash
export OMP_NUM_THREADS=20

for DS in fashionmnist mnist breastmnist; do
  python3 experiments/run_patch_qgan.py \
    --dataset $DS \
    --epochs 50 --batch_size 32 --n_train 5000 \
    --n_critic 3 --eval_every 10 --n_eval 1000 \
    --out_dir ./results --data_root ./data \
    2>&1 | tee results/logs/patchqgan_${DS}.log
done
```

Or use the existing pipeline script (already handles ordering):
```bash
bash experiments/run_patch_pipeline.sh
```

Expected per dataset: ~1.5h (FashionMNIST/MNIST), ~0.5h (BreastMNIST).
Results land in `results/checkpoints/patchqgan_<dataset>/`.

---

## After All Experiments Complete

### 3. Regenerate all paper figures with complete data

```bash
python3 experiments/generate_figures.py
```

This produces `results/figures/fig1_*.pdf` through `fig7_*.pdf`.
Figures that need Patch QGAN data: fig2 (ablation bars), fig3 (cross-dataset),
fig5 (summary table). The expressibility figures (fig6, fig7) are already done.

### 4. Populate remaining cells in LaTeX Table 1

Edit `paper/src/main.tex`, Table `tab:main_results`.
Cells marked `$\star$` need replacing with actual numbers from:
- `results/checkpoints/mnist_no_entanglement/metrics_epoch050.json` → MNIST / no_ent FID, IS
- `results/checkpoints/patchqgan_fashionmnist/metrics_epoch050.json` → FashionMNIST / Patch QGAN
- `results/checkpoints/patchqgan_mnist/metrics_epoch050.json`        → MNIST / Patch QGAN
- `results/checkpoints/patchqgan_breastmnist/metrics_epoch050.json`  → BreastMNIST / Patch QGAN

### 5. Compile the paper

```bash
cd paper/src
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or check if `latexmk` is available: `latexmk -pdf main.tex`

### 6. Final push

```bash
git add results/ paper/src/main.tex
git commit -m "Add Patch QGAN + mnist_no_ent results; final paper figures"
git push origin main
```

---

## What's Already Done

| Item | Status |
|---|---|
| fashionmnist / full | ✅ FID=192.6, IS=1.63, P=0.253, R=0.000 |
| fashionmnist / no_entanglement | ✅ FID=173.7, IS=1.91, P=0.215, R=0.000 |
| fashionmnist / no_style | ✅ FID=219.5, IS=2.80, P=0.022, R=0.001 |
| mnist / full | ✅ FID=118.2, IS=1.79, P=0.149, R=0.005 |
| breastmnist / full | ✅ FID=226.8, IS=1.66, P=0.000, R=0.000 |
| fashionmnist / DCGAN | ✅ FID=55.1, IS=2.46, P=0.303, R=0.235 |
| fashionmnist / WGAN-GP | ✅ FID=296.7 (unstable) |
| mnist / DCGAN | ✅ FID=21.3, IS=1.84 |
| mnist / WGAN-GP | ✅ FID=166.1 (unstable) |
| breastmnist / DCGAN | ✅ FID=170.5, IS=1.57 |
| breastmnist / WGAN-GP | ✅ FID=348.9 (unstable) |
| Expressibility analysis | ✅ KL: full=0.0025, no_ent=0.0003, no_style=1.41, patch=0.0076 |
| Entanglement entropy | ✅ encoding=1.835±0.222 bits, full=1.935±0.042 bits |
| Figures 1–7 (partial) | ✅ Generated, will be updated after Patch QGAN |
| Paper draft (main.tex) | ✅ Complete with real numbers (Patch QGAN cells still `$\star$`) |

---

## Key Results Summary (for context)

The central finding is that the **no-entanglement variant consistently
outperforms the full EQ-GAN**. This is explained by:
1. The no-entanglement circuit is 8× MORE expressive (KL=0.0003 vs 0.0025),
   pushing the full circuit toward a barren plateau regime.
2. The encoding layers alone generate 1.835 bits of entanglement (out of max 2),
   making the explicit CNOT entanglement core redundant as an information channel
   while adding harmful circuit complexity.

The paper is framed as a **negative result investigation**, which is publishable
and valuable for the quantum ML community. Target venue: Quantum Machine
Intelligence (Springer) or Quantum Information Processing.
