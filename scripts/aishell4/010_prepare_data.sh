#!/usr/bin/env bash
stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/AISHELL-4
DATA_DIR=data/aishell4
EXP_DIR=exp/aishell4

mkdir -p exp

if [ $stage -le 0 ]; then
  echo "Preparing AISHELL-4 data..."
  $train_cmd $EXP_DIR/log/prepare.log \
    python local/prepare_aishell4.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

exit 0
