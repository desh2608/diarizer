#!/usr/bin/env bash
stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/AliMeeting
DATA_DIR=data/alimeeting
EXP_DIR=exp/alimeeting

mkdir -p exp

if [ $stage -le 0 ]; then
  echo "Preparing AliMeeting data..."
  $train_cmd $EXP_DIR/log/prepare.log \
    python local/prepare_alimeeting.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

exit 0
