#!/usr/bin/env bash
stage=0
nj=10

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/fisher
EXP_DIR=exp/fisher

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Running spectral clustering on $part..."
    (
    utils/queue.pl --mem 2G JOB=1:$nj $EXP_DIR/$part/log/spectral/sc.JOB.log \
      python diarizer/spectral/sclust_batch.py \
        --in-file-list exp/list_${part}.JOB.txt \
        --out-rttm-dir $EXP_DIR/$part/spectral \
        --xvec-ark-dir $EXP_DIR/$part/xvec \
        --segments-dir $EXP_DIR/$part/xvec \
        --xvec-transform diarizer/models/ResNet101_8kHz/transform.h5 \
        --num-speakers 2 &
    wait
    )
  done
fi

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Evaluating $part (full)"
    cat $DATA_DIR/$part/rttm/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$part/spectral/*.rttm > exp/hyp.rttm
    LC_ALL= spyder exp/ref.rttm exp/hyp.rttm

    echo "Evaluating $part (fair)"
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm -c 0.25 |\
      awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/,/SPEAKER ERROR TIME/,/OVERALL SPEAKER DIARIZATION ERROR/)'
  done
fi

exit 0
