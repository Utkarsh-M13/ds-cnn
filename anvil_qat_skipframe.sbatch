#!/bin/bash
#SBATCH --job-name=kws_qat_skipframe
#SBATCH --account=cis260440-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=qat_skipframe_%j.log

set -uo pipefail

# Prebuilt Python 3.12 env with cmsisdsp + tensorflow + tfmot. Create once with:
#   module load anaconda
#   conda create -y -p "$HOME/qatenv" python=3.12
#   source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$HOME/qatenv"
#   pip install cmsisdsp numpy soundfile pyyaml scipy datasets "tensorflow[and-cuda]>=2.16" tensorflow-model-optimization tf_keras
module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"
export TF_USE_LEGACY_KERAS=1   # tensorflow-model-optimization (QAT) needs Keras 2

cd "$SLURM_SUBMIT_DIR"

# 1. Dataset (skips if already arranged into processed/<label>/*.wav)
if [ ! -d processed ]; then
  python fetch_dataset.py --out processed
fi

# 2. Skip-frame (FrameSkip) variant: DS-CNN width 64, depth 4, MFCC stride 40 ms
#    (~50 frames instead of 99), the lowest-latency variant. No trained float
#    checkpoint exists for it, so train fresh then QAT fine-tune in one run.
#    Recipe matches Bisti's 87.69% skip-frame run (runs/skip_frame on her main):
#    80 epochs + mixup 0.2. Builds a 50-frame cache, tags artifacts "skipframe".
#    Output (the deployable QAT model): kws_tf_skipframe_int8.tflite
python src/train_tf.py --arch dscnn --width 64 --depth 4 \
  --frame-stride-ms 40 --epochs 80 --mixup 0.2 --qat \
  --rebuild-cache --cache mfcc_cache_skipframe.npz --tag skipframe
