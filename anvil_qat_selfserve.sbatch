#!/bin/bash
#SBATCH --job-name=kws_qat_selfserve
#SBATCH --account=cis260440-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=qat_selfserve_%j.log

set -uo pipefail
module load modtree/gpu
module load anaconda

# One-time env (safe to re-run). cmsisdsp builds a native extension and is REQUIRED for the
# arm_f32 features the checkpoint was trained on.
pip install --user "tensorflow[and-cuda]>=2.16" tensorflow-model-optimization \
    datasets soundfile numpy pyyaml cmsisdsp scipy

cd "$SLURM_SUBMIT_DIR"   # submit from inside the ds-cnn folder

# 1. Download + arrange the dataset into processed/<label>/*.wav (skips if already there)
if [ ! -d processed ]; then
  python fetch_dataset.py --out processed
fi

# 2. Build arm_f32 features and run the QAT fine-tune from the existing checkpoint
python src/train_tf.py --init kws_tf.keras --epochs 0 --qat --rebuild-cache
