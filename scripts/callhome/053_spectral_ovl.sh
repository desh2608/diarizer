#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/callhome_2spk
EXP_DIR=exp/callhome_2spk

mkdir -p exp

if [ $stage -le 0 ]; then
  for split in dev test; do
    echo "Running spectral clustering on $split..."
    (
    for audio in $(ls $DATA_DIR/${split}/audios_8k/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      utils/queue.pl --mem 2G -l hostname="!b03*" $EXP_DIR/$split/log/spectral_ovl/sc_${filename}.log \
        python diarizer/spectral/sclust.py \
          --out-rttm-dir $EXP_DIR/$split/spectral_ovl \
          --xvec-ark-file $EXP_DIR/$split/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$split/xvec/${filename}.seg \
          --overlap-rttm $EXP_DIR/$split/ovl/${filename}.rttm \
          --xvec-transform diarizer/models/ResNet101_8kHz/transform.h5 \
          --num-speakers 2 \
          --max-neighbors 30 &
    done
    wait
    )
  done
fi

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Evaluating $part (full)"
    cat $DATA_DIR/$part/rttm/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$part/spectral_ovl/*.rttm > exp/hyp.rttm
    LC_ALL= spyder exp/ref.rttm exp/hyp.rttm

    echo "Evaluating $part (fair)"
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm -c 0.25 |\
      awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/,/SPEAKER ERROR TIME/,/OVERALL SPEAKER DIARIZATION ERROR/)'
  done
fi

exit 0
