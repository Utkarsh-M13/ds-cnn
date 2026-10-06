#!/bin/bash
#SBATCH --job-name=kws_qat_selfserve
#SBATCH --account=cis260440-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=qat_selfserve_%j.log

set -uo pipefail

# Use the prebuilt Python 3.12 env (~/qatenv) that has cmsisdsp + tensorflow + tfmot + deps.
# Create it once on the login node with:
#   module load anaconda
#   conda create -y -p ~/qatenv python=3.12
#   source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ~/qatenv
#   pip install cmsisdsp numpy soundfile pyyaml scipy datasets "tensorflow[and-cuda]>=2.16" tensorflow-model-optimization
module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"
export TF_USE_LEGACY_KERAS=1   # tensorflow-model-optimization (QAT) needs Keras 2, not Keras 3

cd "$SLURM_SUBMIT_DIR"   # run from inside the ds-cnn folder

# 1. Download + arrange the dataset into processed/<label>/*.wav (skips if already there)
if [ ! -d processed ]; then
  python fetch_dataset.py --out processed
fi

# 2. Build arm_f32 features and run the QAT fine-tune from the existing checkpoint
python src/train_tf.py --init kws_tf.keras --epochs 0 --qat --rebuild-cache
