#!/usr/bin/env bash
stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/alimeeting
EXP_DIR=exp/alimeeting

mkdir -p exp

if [ $stage -le 0 ]; then
  for split in eval test; do
    echo "Running spectral clustering on $split..."
    (
    for audio in $(ls $DATA_DIR/${split}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      $train_cmd $EXP_DIR/$split/log/spectral_ovl/sc_${filename}.log \
        python diarizer/spectral/sclust.py \
          --out-rttm-dir $EXP_DIR/$split/spectral_ovl \
          --xvec-ark-file $EXP_DIR/$split/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$split/xvec/${filename}.seg \
          --overlap-rttm $EXP_DIR/$split/ovl/${filename}.rttm \
          --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 &
    done
    wait
    )
  done
fi

if [ $stage -le 1 ]; then
  for split in eval test; do
    echo "Evaluating $split"
    cat $DATA_DIR/$split/rttm/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$split/spectral_ovl/*.rttm > exp/hyp.rttm
    ./md-eval.pl -c 0.25 -r exp/ref.rttm -s exp/hyp.rttm
  done
fi

exit 0
