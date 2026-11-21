#!/bin/bash
#SBATCH --job-name=kws_acc_k5_r3
#SBATCH --account=cis260440-ai
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --array=0-1
#SBATCH --output=acc_k5_r3_%A_%a.log

# Round 3 of the K5 accuracy experiments (13 MFCC x 49 frames, width 64, depth 4,
# 120 ms NN on the nRF5340; none of these change the deployed graph).
# Round 2 best: cos200 + noise = 89.88% post-QAT. cos400 alone = 89.48% (+0.8 over cos200).
#
#   task 0  k5_cos400_noise    : cosine 400 epochs + background noise. Do they stack again?
#   task 1  k5_cos400_noise_kd : task 0 + distillation from the c48 baseline
#                                (w128d6, 48 MFCC, 93.78% post-QAT). alpha 0.5, T 4.
#                                Clean A/B against task 0: only KD differs.
#
# KD is float training only. The teacher reads all 48 MFCCs, the student reads
# the first 13 of the same cache (identical to the c13 cache). Task 1 builds the
# c48 noise cache once. Only one task may build it: np.savez is not atomic.
# QAT runs without the teacher on the existing c13 noise cache, as in round 2.
#
# Wall time: limits are ~1.5x the expected run time.
#   task 0: 200 epochs + noise took 4:00 in round 2, so 400 epochs ~ 6:40.
#   task 1: + c48 noise cache build (~3-5 min) + teacher forward pass each step ~ 15 min.
# Submit each task with its own limit, so task 0 does not hold 25 min:
#   sbatch --array=0 --time=00:10:00 anvil_acc_k5_r3_v2.sbatch
#   sbatch --array=1 --time=00:25:00 anvil_acc_k5_r3_v2.sbatch
#
# Smoke test on the login node (no queue), both tasks:
#   for t in 0 1; do SMOKE=1 SLURM_ARRAY_TASK_ID=$t SLURM_SUBMIT_DIR=$PWD bash anvil_acc_k5_r3_v2.sbatch > smoke_r3_$t.log 2>&1 && echo "$t OK" || echo "$t FAILED"; done
#   (The task 1 smoke builds the c48 noise cache, so the real task 1 skips that step.)
# Fallback queue:
#   sbatch -A cis260440-gpu -p gpu --array=0 --time=00:10:00 anvil_acc_k5_r3_v2.sbatch
# Results:
#   grep -HE "float accuracy|post-QAT accuracy" acc_k5_r3_*.log
#   sacct -j <jobid> --format=JobID,State,Elapsed,Timelimit

set -euo pipefail

module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"

cd "$SLURM_SUBMIT_DIR"

BASE=(--num-mfcc 13 --frame-stride-ms 20)
NOISE_DIR="${NOISE_DIR:-downloads/speech_commands_v0.02/_background_noise_}"
TEACHER="${TEACHER:-runs/baseline_c48_v2/kws_tf.keras}"
FEAT=(--noise-aug --noise-copies 1 --snr-min 0 --snr-max 15 --noise-dir "$NOISE_DIR")
RECIPE=(--epochs 400 --lr-schedule cosine --warmup-epochs 5)
KD=()       # float training only

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) RUN=acc_k5_cos400_noise ;;
  1) RUN=acc_k5_cos400_noise_kd; KD=(--kd-teacher "$TEACHER" --kd-alpha 0.5 --kd-temp 4) ;;
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
echo "[job] task=${SLURM_ARRAY_TASK_ID:-0} run=$RUN feat=${BASE[*]} ${FEAT[*]} recipe=${RECIPE[*]} ${KD[*]:-}"

if ! ls "$NOISE_DIR"/*.wav >/dev/null 2>&1; then
  echo "[error] no background-noise wavs in $NOISE_DIR."
  echo "        Set NOISE_DIR=/path/to/_background_noise_ (sbatch --export=ALL,NOISE_DIR=...)."
  exit 1
fi
if [[ ${#KD[@]} -gt 0 ]] && [[ ! -e "$TEACHER" || ! -f "${TEACHER%.keras}.json" ]]; then
  echo "[error] teacher $TEACHER or its .json is missing. It must be the FLOAT c48 model"
  echo "        (Keras 3), not the QAT one. Set TEACHER=... (sbatch --export=ALL,TEACHER=...)."
  exit 1
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  env -u TF_USE_LEGACY_KERAS python src/train_tf.py \
    --arch dscnn --width 64 --depth 4 "${BASE[@]}" "${FEAT[@]}" "${RECIPE[@]}" \
    ${KD[@]+"${KD[@]}"} --mixup 0.2 --label-smoothing 0.05 --smoke --run-name "${RUN}_smoke"
  exit 0
fi

# 1) Float training under Keras 3 (with the teacher for task 1).
env -u TF_USE_LEGACY_KERAS python src/train_tf.py \
  --arch dscnn --width 64 --depth 4 "${BASE[@]}" "${FEAT[@]}" "${RECIPE[@]}" \
  ${KD[@]+"${KD[@]}"} --mixup 0.2 --label-smoothing 0.05 \
  --run-name "$RUN"

# 2) QAT under Keras 2, from a version-agnostic weights bundle (no teacher).
env -u TF_USE_LEGACY_KERAS python tools/export_weights.py "runs/$RUN/kws_tf.keras"
export TF_USE_LEGACY_KERAS=1
python src/train_tf.py --init "runs/$RUN/kws_tf.keras" --epochs 0 \
  "${BASE[@]}" "${FEAT[@]}" --qat --run-name "${RUN}_qat"
