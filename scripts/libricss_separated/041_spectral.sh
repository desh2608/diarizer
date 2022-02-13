#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/libricss_separated_oracle
EXP_DIR=exp/libricss_separated_oracle

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Running spectral clustering on ${part}..."
    (
    while read -r line
    do
      filename=$(echo $line | cut -d' ' -f1)

      # Here we combine x-vectors of one recording into 1 ark and segments file, and modify the segments
      # file recording ids to be the same as the recording ids (removing last 2 characters).

      cat $EXP_DIR/$part/xvec/${filename}_{0,1}.ark > $EXP_DIR/$part/xvec/$filename.ark
      cat $EXP_DIR/$part/xvec/${filename}_{0,1}.seg | awk '{print $1, substr($2, 1, length($2)-2), $3, $4}' > $EXP_DIR/$part/xvec/$filename.seg

      utils/queue.pl --mem 2G -l hostname="!b03*" $EXP_DIR/${part}/log/spectral/spectral_${filename}.log \
        python diarizer/spectral/sclust2.py \
            --out-rttm-dir $EXP_DIR/${part}/spectral \
            --xvec-ark-file $EXP_DIR/${part}/xvec/${filename}.ark \
            --segments-file $EXP_DIR/${part}/xvec/${filename}.seg \
            --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 \
            --max-neighbors 30 &
    done<$DATA_DIR/${part}/reco2channel
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
