# Print immediately
export PYTHONUNBUFFERED=1

# Activate environment
. /home/draj/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate vbx

export PATH=${PATH}:`pwd`/utils
export LC_ALL=C
