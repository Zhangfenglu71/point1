#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline for the "clean, self-consistent" E6 baseline:
# train (FiLM + CFG-training, velocity RF) -> sample (Velocity-CFG w=3) -> eval (classifier ACC)
#
# Assumptions:
# - Run from repo root (the folder that contains `scripts/`, `models/`, `checkpoints/`, `data/`)
# - Python env has required deps (torch, torchvision, opencv-python, etc.)
#
# Outputs:
# - checkpoint: checkpoints/video_cond_flow_film_cfg_best.pth
# - samples:    samples_baseline_e6_w3/...
# - meta:       samples_baseline_e6_w3/meta.json
#
# You can override key knobs via env vars, e.g.:
#   NUM_STEPS=40 SEED=123 ALPHA=0.25 bash run_e6_pipeline.sh

NUM_STEPS="${NUM_STEPS:-50}"
SEED="${SEED:-0}"
ALPHA="${ALPHA:-0.3}"
GRID_N_PER_ACTION="${GRID_N_PER_ACTION:-4}"
EVAL_N_PER_ACTION="${EVAL_N_PER_ACTION:-28}"

CKPT="${CKPT:-checkpoints/video_cond_flow_film_cfg_best.pth}"
SAMPLES_DIR="${SAMPLES_DIR:-samples_baseline_e6_w3}"
CLS_CKPT="${CLS_CKPT:-checkpoints/radar_cls_resnet18_best.pth}"

LOG_DIR="${LOG_DIR:-runs/e6_pipeline_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

echo "[run_e6_pipeline] log_dir=${LOG_DIR}"
echo "[run_e6_pipeline] num_steps=${NUM_STEPS} seed=${SEED} alpha=${ALPHA} grid_n=${GRID_N_PER_ACTION} eval_n=${EVAL_N_PER_ACTION}"
echo "[run_e6_pipeline] ckpt=${CKPT}"
echo "[run_e6_pipeline] samples_dir=${SAMPLES_DIR}"
echo "[run_e6_pipeline] cls_ckpt=${CLS_CKPT}"
echo

# 1) Train (FiLM + CFG-training, velocity RF)
echo "[1/3] Train: scripts.train_video_cond_flow_film_cfg"
python -m scripts.train_video_cond_flow_film_cfg 2>&1 | tee "${LOG_DIR}/train.log"
echo

# 2) Sample (Velocity-CFG w=3 is fixed inside sample_baseline_e6_w3.py)
echo "[2/3] Sample: scripts.sample_baseline_e6_w3"
python -m scripts.sample_baseline_e6_w3 \
  --num_steps "${NUM_STEPS}" \
  --seed "${SEED}" \
  --alpha "${ALPHA}" \
  --grid_n_per_action "${GRID_N_PER_ACTION}" \
  --eval_n_per_action "${EVAL_N_PER_ACTION}" \
  --ckpt "${CKPT}" 2>&1 | tee "${LOG_DIR}/sample.log"
echo

# 3) Eval
echo "[3/3] Eval: scripts.eval_gen_with_cls"
python -m scripts.eval_gen_with_cls \
  --root "${SAMPLES_DIR}" \
  --ckpt "${CLS_CKPT}" 2>&1 | tee "${LOG_DIR}/eval.log"
echo

echo "[run_e6_pipeline] DONE"
echo "[run_e6_pipeline] train_log=${LOG_DIR}/train.log"
echo "[run_e6_pipeline] sample_log=${LOG_DIR}/sample.log"
echo "[run_e6_pipeline] eval_log=${LOG_DIR}/eval.log"
