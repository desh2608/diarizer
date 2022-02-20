#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/LibriCSS
# SEPARATED_DIR=/export/c03/zhuc/css
SEPARATED_DIR=data/libricss_separated_oracle/wav/ # for oracle
DATA_DIR=data/libricss_separated_oracle
EXP_DIR=exp/libricss_separated_oracle

mkdir -p exp

# NOTE: Please modify the `file_name_to_session_and_channel()` function in the following
# script to suit your own wav naming convention. Currently, it supports names of the
# form `overlap_ratio_10.0_sil0.1_1.0_session1_actual10.2_channel_1.wav`.

if [ $stage -le 0 ]; then
  echo "Preparing separated LibriCSS data..."
  python local/prepare_libricss_separated.py --oracle --data-dir $CORPUS_DIR --output-dir $DATA_DIR --separated-dir $SEPARATED_DIR
fi

exit 0
