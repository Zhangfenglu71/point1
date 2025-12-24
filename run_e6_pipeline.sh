#!/usr/bin/env bash
set -euo pipefail

NUM_STEPS="${NUM_STEPS:-50}"
SEED="${SEED:-0}"
ALPHA="${ALPHA:-0.3}"

DO_TRAIN="${DO_TRAIN:-1}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts.train_video_cond_flow_film_cfg}"   # or scripts.train_video_cond_flow_film_cfg_deltavar
CFG_W="${CFG_W:-3}"

CKPT="${CKPT:-checkpoints/video_cond_flow_film_cfg_best.pth}"
CLS_CKPT="${CLS_CKPT:-checkpoints/radar_cls_resnet18_best.pth}"

OUT_DIR="${OUT_DIR:-tmp_run}"
DELTA_STATS_OUT="${DELTA_STATS_OUT:-delta.csv}"

LOG_DIR="${LOG_DIR:-runs/pipeline_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_DIR}"

echo "[pipeline] log_dir=${LOG_DIR}"
echo "[pipeline] DO_TRAIN=${DO_TRAIN} TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[pipeline] ckpt=${CKPT}"
echo "[pipeline] out_dir=${OUT_DIR}"
echo "[pipeline] num_steps=${NUM_STEPS} seed=${SEED} alpha=${ALPHA} cfg_w=${CFG_W}"
echo

# 1) Train
if [ "${DO_TRAIN}" = "1" ]; then
  echo "[1/3] Train: ${TRAIN_SCRIPT}"
  python -m "${TRAIN_SCRIPT}" 2>&1 | tee "${LOG_DIR}/train.log"
  echo
else
  echo "[1/3] Train: skipped (DO_TRAIN=0)"
  echo
fi

# 2) Sample
echo "[2/3] Sample: scripts.sample_baseline_e6_w3_delta"
python -m scripts.sample_baseline_e6_w3_delta \
  --model_type film \
  --ckpt "${CKPT}" \
  --base_ch 64 \
  --cfg_w "${CFG_W}" \
  --num_steps "${NUM_STEPS}" \
  --seed "${SEED}" \
  --alpha "${ALPHA}" \
  --log_delta_stats \
  --delta_stats_out "${DELTA_STATS_OUT}" \
  --out_dir "${OUT_DIR}" 2>&1 | tee "${LOG_DIR}/sample.log"
echo

# 3) Eval
echo "[3/3] Eval: scripts.eval_gen_with_cls"
python -m scripts.eval_gen_with_cls \
  --root "${OUT_DIR}" \
  --ckpt "${CLS_CKPT}" 2>&1 | tee "${LOG_DIR}/eval.log"
echo

echo "[pipeline] DONE"
echo "[pipeline] train_log=${LOG_DIR}/train.log"
echo "[pipeline] sample_log=${LOG_DIR}/sample.log"
echo "[pipeline] eval_log=${LOG_DIR}/eval.log"
