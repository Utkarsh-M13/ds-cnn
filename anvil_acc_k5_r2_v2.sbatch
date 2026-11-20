#!/bin/bash
#SBATCH --job-name=kws_acc_k5_r2
#SBATCH --account=cis260440-ai
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-3
#SBATCH --output=acc_k5_r2_%A_%a.log

# Round 2 of the K5 accuracy experiments (13 MFCC x 49 frames, width 64, depth 4,
# 120 ms NN on the nRF5340; none of these change the deployed graph).
# Round 1 winner: cosine 200 epochs, 5 warmup = 88.68% post-QAT (K5 86.25%).
# Round 1 also: noise +0.5, weight decay no gain, 8 kHz -1.1.
#
#   task 0  k5_cos200_noise : cos200 + background noise (1 copy, SNR 0-15 dB). Do the gains stack?
#   task 1  k5_cos400       : cosine over 400 epochs. Does even longer help?
#   task 2  k5_cos200_s1    : cos200 with train seed 1  } true repeats: new weight init
#   task 3  k5_cos200_s2    : cos200 with train seed 2  } and shuffle, same data split
#
# Tasks 2 and 3 plus round 1's cos200 (config seed) give 3 runs of the winner.
# Round 1's K5 repeat kept the same seed, so it only measured GPU randomness.
#
# Wall time: round 1 cos200 took 2:40 with QAT; 400 epochs ~5 min. Task 0 reuses
# round 1's noise cache. 20 min leaves margin.
#
# Smoke test on the login node (no queue), all tasks:
#   for t in 0 1 2 3; do SMOKE=1 SLURM_ARRAY_TASK_ID=$t SLURM_SUBMIT_DIR=$PWD bash anvil_acc_k5_r2_v2.sbatch > smoke_r2_$t.log 2>&1 && echo "$t OK" || echo "$t FAILED"; done
# Fallback queue:
#   sbatch -A cis260440-gpu -p gpu anvil_acc_k5_r2_v2.sbatch
# Results:
#   grep -HE "float accuracy|post-QAT accuracy" acc_k5_r2_*.log

set -euo pipefail

module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"

cd "$SLURM_SUBMIT_DIR"

BASE=(--num-mfcc 13 --frame-stride-ms 20)
NOISE_DIR="${NOISE_DIR:-downloads/speech_commands_v0.02/_background_noise_}"
COS200=(--epochs 200 --lr-schedule cosine --warmup-epochs 5)
FEAT=()     # feature flags: used by float training AND QAT
SEED=()     # train seed: used by float training AND QAT
RECIPE=("${COS200[@]}")

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) RUN=acc_k5_cos200_noise; FEAT=(--noise-aug --noise-copies 1 --snr-min 0 --snr-max 15 --noise-dir "$NOISE_DIR") ;;
  1) RUN=acc_k5_cos400;       RECIPE=(--epochs 400 --lr-schedule cosine --warmup-epochs 5) ;;
  2) RUN=acc_k5_cos200_s1;    SEED=(--train-seed 1) ;;
  3) RUN=acc_k5_cos200_s2;    SEED=(--train-seed 2) ;;
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
echo "[job] task=${SLURM_ARRAY_TASK_ID:-0} run=$RUN feat=${BASE[*]} ${FEAT[*]:-} recipe=${RECIPE[*]} ${SEED[*]:-}"

if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]] && ! ls "$NOISE_DIR"/*.wav >/dev/null 2>&1; then
  echo "[error] no background-noise wavs in $NOISE_DIR."
  echo "        Set NOISE_DIR=/path/to/_background_noise_ (sbatch --export=ALL,NOISE_DIR=...)."
  exit 1
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  env -u TF_USE_LEGACY_KERAS python src/train_tf.py \
    --arch dscnn --width 64 --depth 4 "${BASE[@]}" ${FEAT[@]+"${FEAT[@]}"} "${RECIPE[@]}" \
    ${SEED[@]+"${SEED[@]}"} --mixup 0.2 --label-smoothing 0.05 --smoke --run-name "${RUN}_smoke"
  exit 0
fi

# 1) Float training under Keras 3.
env -u TF_USE_LEGACY_KERAS python src/train_tf.py \
  --arch dscnn --width 64 --depth 4 "${BASE[@]}" ${FEAT[@]+"${FEAT[@]}"} "${RECIPE[@]}" \
  ${SEED[@]+"${SEED[@]}"} --mixup 0.2 --label-smoothing 0.05 \
  --run-name "$RUN"

# 2) QAT under Keras 2, from a version-agnostic weights bundle (as in the K4/K5 job).
env -u TF_USE_LEGACY_KERAS python tools/export_weights.py "runs/$RUN/kws_tf.keras"
export TF_USE_LEGACY_KERAS=1
python src/train_tf.py --init "runs/$RUN/kws_tf.keras" --epochs 0 \
  "${BASE[@]}" ${FEAT[@]+"${FEAT[@]}"} ${SEED[@]+"${SEED[@]}"} --qat --run-name "${RUN}_qat"
