# Print immediately
export PYTHONUNBUFFERED=1

export KALDI_ROOT=/export/b16/draj/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

# Activate environment
. /home/draj/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate vbx

export LC_ALL=C
