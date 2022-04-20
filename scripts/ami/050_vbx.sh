#!/usr/bin/env bash
stage=0

# Hyperparameters (from original repo)
Fa=0.4
Fb=64
loopP=0.65

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/ami
EXP_DIR=exp/ami

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      $train_cmd $EXP_DIR/$part/log/vbx/vb_${filename}.log \
        python diarizer/vbx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir $EXP_DIR/$part/vbx \
          --xvec-ark-file $EXP_DIR/$part/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$part/xvec/${filename}.seg \
          --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 \
          --plda-file diarizer/models/ResNet101_16kHz/plda \
          --threshold -0.015 \
          --init-smoothing 7.0 \
          --lda-dim 128 \
          --Fa $Fa \
          --Fb $Fb \
          --loopP $loopP &
    done
    wait
    )
  done
fi

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Evaluating $part"
    cat $DATA_DIR/$part/rttm_but/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$part/vbx/*.rttm > exp/hyp.rttm
    LC_ALL= spyder --per-file exp/ref.rttm exp/hyp.rttm
  done
fi

exit 0
