# Print immediately
export PYTHONUNBUFFERED=1

# Activate environment
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda deactivate && conda activate diar

export PATH=${PATH}:`pwd`/utils
export LC_ALL=C
