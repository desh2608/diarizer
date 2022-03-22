#!/usr/bin/env bash
stage=0

# Hyperparameters (from original repo)
Fa=0.4
Fb=17
loopP=0.40

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/callhome_2spk
EXP_DIR=exp/callhome_2spk

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios_8k/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      utils/queue.pl --mem 2G $EXP_DIR/$part/log/vbx/vb_${filename}.log \
        python diarizer/vbx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir $EXP_DIR/$part/vbx \
          --xvec-ark-file $EXP_DIR/$part/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$part/xvec/${filename}.seg \
          --xvec-transform diarizer/models/ResNet101_8kHz/transform.h5 \
          --plda-file diarizer/models/ResNet101_8kHz/plda \
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
