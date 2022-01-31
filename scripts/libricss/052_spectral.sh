#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/LibriCSS
DATA_DIR=data/libricss
EXP_DIR=exp/libricss

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Running spectral clustering on ${part}..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')

      utils/queue.pl --mem 2G -l hostname="!b03*" $EXP_DIR/${part}/log/spectral/spectral_${filename}.log \
        python diarizer/spectral/sclust.py \
            --out-rttm-dir $EXP_DIR/${part}/spectral \
            --xvec-ark-file $EXP_DIR/${part}/xvec/${filename}.ark \
            --segments-file $EXP_DIR/${part}/xvec/${filename}.seg \
            --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 &
    done
    wait
    )
  done
fi

if [ $stage -le 1 ]; then
  # Combine all RTTM files and score
  for part in dev test; do
    cat $DATA_DIR/${part}/rttm/*.rttm > $EXP_DIR/ref.rttm
    cat $EXP_DIR/${part}/spectral/*.rttm > $EXP_DIR/hyp.rttm
    LC_ALL= spyder --per-file $EXP_DIR/ref.rttm $EXP_DIR/hyp.rttm
  done
fi

exit 0
