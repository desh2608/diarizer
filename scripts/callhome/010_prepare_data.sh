#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/corpora5/LDC/LDC2001S97
DATA_DIR=data/callhome
EXP_DIR=exp/callhome

mkdir -p exp

if [ $stage -le 0 ]; then
  python local/prepare_callhome.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

exit 0
