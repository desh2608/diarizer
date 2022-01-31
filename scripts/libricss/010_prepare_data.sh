#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/LibriCSS
DATA_DIR=data/libricss
EXP_DIR=exp/libricss

mkdir -p exp

if [ $stage -le 0 ]; then
  echo "Preparing LibriCSS data..."
  python local/prepare_libricss.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

exit 0
