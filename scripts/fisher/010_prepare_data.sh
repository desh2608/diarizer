#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=data/fisher/dl
DATA_DIR=data/fisher
EXP_DIR=exp/fisher

mkdir -p exp

if [ $stage -le 0 ]; then
  echo "Doing nothing since data is already prepared"
fi

exit 0
