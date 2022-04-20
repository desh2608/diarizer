#!/usr/bin/env bash
stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/aishell4
EXP_DIR=exp/aishell4

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Running spectral clustering on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      $train_cmd $EXP_DIR/$part/log/spectral/sc_${filename}.log \
        python diarizer/spectral/sclust.py \
          --out-rttm-dir $EXP_DIR/$part/spectral \
          --xvec-ark-file $EXP_DIR/$part/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$part/xvec/${filename}.seg \
          --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 &
    done
    wait
    )
  done
fi

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Evaluating $part"
    cat $DATA_DIR/$part/rttm/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$part/spectral/*.rttm > exp/hyp.rttm
    LC_ALL= spyder --per-file exp/ref.rttm exp/hyp.rttm
  done
fi

exit 0
