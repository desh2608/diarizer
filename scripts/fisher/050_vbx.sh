#!/usr/bin/env bash
stage=0
nj=10

# Hyperparameters (from original repo)
Fa=0.1
Fb=64
loopP=0.3

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/fisher
EXP_DIR=exp/fisher

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $part..."
    (
    utils/queue.pl --mem 2G JOB=1:$nj $EXP_DIR/$part/log/vbx/vb.JOB.log \
      python diarizer/vbx/vbhmm_batch.py \
        --init AHC+VB \
        --in-file-list exp/list_${part}.JOB.txt \
        --out-rttm-dir $EXP_DIR/$part/vbx \
        --xvec-ark-dir $EXP_DIR/$part/xvec \
        --segments-dir $EXP_DIR/$part/xvec \
        --xvec-transform diarizer/models/ResNet101_8kHz/transform.h5 \
        --plda-file diarizer/models/ResNet101_8kHz/plda \
        --threshold -0.015 \
        --init-smoothing 7.0 \
        --lda-dim 128 \
        --Fa $Fa \
        --Fb $Fb \
        --loopP $loopP &
    wait
    )
  done
fi

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Evaluating $part (full)"
    cat $DATA_DIR/$part/rttm/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$part/vbx/*.rttm > exp/hyp.rttm
    LC_ALL= spyder exp/ref.rttm exp/hyp.rttm

    echo "Evaluating $part (fair)"
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm -c 0.25 |\
      awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/,/SPEAKER ERROR TIME/,/OVERALL SPEAKER DIARIZATION ERROR/)'
  done
fi

exit 0
