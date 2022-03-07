#!/usr/bin/env bash
stage=0

# VBx Hyperparameters (tuned on session0)
Fa=0.1
Fb=5
loopP=0.9

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/LibriCSS
DATA_DIR=data/libricss_separated_v2_multi
EXP_DIR=exp/libricss_separated_v2_multi

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev; do
    echo "Running VBx on ${part} with Fa=$Fa, Fb=$Fb, loopP=$loopP"
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')

      # run variational bayes on top of x-vectors
      utils/queue.pl --mem 2G $EXP_DIR/${part}/log/vbx/vb_${filename}.log \
        python diarizer/vbx/vbhmm.py \
            --init AHC+VB \
            --out-rttm-dir $EXP_DIR/${part}/vbx \
            --xvec-ark-file $EXP_DIR/${part}/xvec/${filename}.ark \
            --segments-file $EXP_DIR/${part}/xvec/${filename}.seg \
            --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 \
            --plda-file diarizer/models/ResNet101_16kHz/plda \
            --threshold -0.015 \
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
  # Combine all RTTM files and score
  for part in dev; do
    cat $DATA_DIR/${part}/rttm/*.rttm > $EXP_DIR/ref.rttm
    cat $EXP_DIR/${part}/vbx/*.rttm > $EXP_DIR/hyp.rttm
    LC_ALL= spyder --per-file $EXP_DIR/ref.rttm $EXP_DIR/hyp.rttm
  done
fi

exit 0
