#!/usr/bin/env bash
stage=0

# Hyperparameters (same as AISHELL-4)
Fa=0.5
Fb=40
loopP=0.9

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/alimeeting
EXP_DIR=exp/alimeeting

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in eval test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      $train_cmd $EXP_DIR/$part/log/vbx_ovl/vb_${filename}.log \
        python diarizer/vbx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir $EXP_DIR/$part/vbx_ovl \
          --xvec-ark-file $EXP_DIR/$part/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$part/xvec/${filename}.seg \
          --overlap-rttm $EXP_DIR/$part/ovl/${filename}.rttm \
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
  for part in eval test; do
    echo "Evaluating $part"
    cat $DATA_DIR/$part/rttm/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$part/vbx_ovl/*.rttm > exp/hyp.rttm
    ./md-eval.pl -c 0.25 -r exp/ref.rttm -s exp/hyp.rttm
  done
fi

exit 0
