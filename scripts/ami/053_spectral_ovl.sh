#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/corpora5/amicorpus
DATA_DIR=data/ami
EXP_DIR=exp/ami

mkdir -p exp

if [ $stage -le 0 ]; then
  for split in dev test; do
    echo "Running spectral clustering on $split..."
    (
    for audio in $(ls $DATA_DIR/${split}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      utils/queue.pl --mem 2G -l hostname="!b03*" $EXP_DIR/$split/log/spectral_ovl/sc_${filename}.log \
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
  for split in dev test; do
    echo "Evaluating $split"
    cat $DATA_DIR/$split/rttm_but/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$split/spectral_ovl/*.rttm > exp/hyp.rttm
    LC_ALL= spyder --per-file exp/ref.rttm exp/hyp.rttm
  done
fi

exit 0
