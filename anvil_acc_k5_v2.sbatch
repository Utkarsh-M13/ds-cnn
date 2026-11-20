#!/bin/bash
#SBATCH --job-name=kws_acc_k5
#SBATCH --account=cis260440-ai
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-5
#SBATCH --output=acc_k5_%A_%a.log

# Accuracy experiments on the K5 shape (13 MFCC x 49 frames, width 64, depth 4).
# K5 runs at 120 ms NN time on the nRF5340. None of these change the deployed
# graph, so latency stays the same. Each task changes ONE thing from the K5
# recipe (80 epochs, Adam 1e-3, plateau LR, mixup 0.2, label smoothing 0.05),
# then runs QAT and exports int8. Reference: K5 post-QAT 86.25%.
#
#   task 0  k5_sr8k        : resample to 8 kHz before MFCC (mel range 20-3800 Hz).
#                            Bisti's v1 sr8k run gave +2.7 over w64 at 16 kHz.
#                            NOTE: firmware must also decimate to 8 kHz to deploy.
#   task 1  k5_cos200      : 200 epochs, 5 warmup, cosine decay (no early stop)
#   task 2  k5_cos200_wd1  : task 1 + AdamW weight decay 0.01
#   task 3  k5_cos200_wd5  : task 1 + AdamW weight decay 0.05
#   task 4  k5_noise       : + background-noise mixing (1 noisy copy, SNR 0-15 dB)
#   task 5  k5_repeat      : exact K5 recipe again, to measure run-to-run noise.
#                            Treat any gain smaller than |task 5 - 86.25| as noise.
#
# Wall time: K5 float was ~1 s/epoch, QAT ~2 min. 200 epochs + QAT ~7 min.
# Tasks 0 and 4 build new MFCC caches first (+2 to 5 min). 20 min leaves margin.
# Caches have distinct names (sample rate, noise settings), so tasks never
# write the same file. Tasks 1, 2, 3 and 5 reuse the K5 16 kHz cache.
#
# Smoke test first (1 epoch on a tiny subset, no export, all tasks):
#   sbatch --export=ALL,SMOKE=1 --time=00:08:00 anvil_acc_k5_v2.sbatch
# Fallback queue:
#   sbatch -A cis260440-gpu -p gpu anvil_acc_k5_v2.sbatch
# Results:
#   grep -HE "float accuracy|post-QAT accuracy" acc_k5_*.log
#
# PREREQUISITE: processed/ holds the v2 wavs. Task 4 also needs the Speech
# Commands background-noise wavs in $NOISE_DIR (checked below).

set -euo pipefail

module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"

cd "$SLURM_SUBMIT_DIR"

BASE=(--num-mfcc 13 --frame-stride-ms 20)
NOISE_DIR="${NOISE_DIR:-downloads/speech_commands_v0.02/_background_noise_}"
FEAT=()     # feature flags: used by float training AND QAT
RECIPE=(--epochs 80)

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) RUN=acc_k5_sr8k;       FEAT=(--sample-rate 8000) ;;
  1) RUN=acc_k5_cos200;     RECIPE=(--epochs 200 --lr-schedule cosine --warmup-epochs 5) ;;
  2) RUN=acc_k5_cos200_wd1; RECIPE=(--epochs 200 --lr-schedule cosine --warmup-epochs 5 --weight-decay 0.01) ;;
  3) RUN=acc_k5_cos200_wd5; RECIPE=(--epochs 200 --lr-schedule cosine --warmup-epochs 5 --weight-decay 0.05) ;;
  4) RUN=acc_k5_noise;      FEAT=(--noise-aug --noise-copies 1 --snr-min 0 --snr-max 15 --noise-dir "$NOISE_DIR") ;;
  5) RUN=acc_k5_repeat ;;
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
echo "[job] task=${SLURM_ARRAY_TASK_ID:-0} run=$RUN feat=${BASE[*]} ${FEAT[*]:-} recipe=${RECIPE[*]}"

if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "4" ]] && ! ls "$NOISE_DIR"/*.wav >/dev/null 2>&1; then
  echo "[error] no background-noise wavs in $NOISE_DIR."
  echo "        Set NOISE_DIR=/path/to/_background_noise_ (sbatch --export=ALL,NOISE_DIR=...)."
  exit 1
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  env -u TF_USE_LEGACY_KERAS python src/train_tf.py \
    --arch dscnn --width 64 --depth 4 "${BASE[@]}" ${FEAT[@]+"${FEAT[@]}"} "${RECIPE[@]}" \
    --mixup 0.2 --label-smoothing 0.05 --smoke --run-name "${RUN}_smoke"
  exit 0
fi

# 1) Float training under Keras 3.
env -u TF_USE_LEGACY_KERAS python src/train_tf.py \
  --arch dscnn --width 64 --depth 4 "${BASE[@]}" ${FEAT[@]+"${FEAT[@]}"} "${RECIPE[@]}" \
  --mixup 0.2 --label-smoothing 0.05 \
  --run-name "$RUN"

# 2) QAT under Keras 2, from a version-agnostic weights bundle (as in the K4/K5 job).
env -u TF_USE_LEGACY_KERAS python tools/export_weights.py "runs/$RUN/kws_tf.keras"
export TF_USE_LEGACY_KERAS=1
python src/train_tf.py --init "runs/$RUN/kws_tf.keras" --epochs 0 \
  "${BASE[@]}" ${FEAT[@]+"${FEAT[@]}"} --qat --run-name "${RUN}_qat"
