#!/bin/bash

DATASET=AMI
EXP_DIR=exp/pyannote/ami

. ./path.sh
. ./utils/parse_options.sh

export PYANNOTE_DATABASE_CONFIG=local/pyannote/database.yml

mkdir -p $EXP_DIR

utils/queue-freegpu.pl -v PYANNOTE_DATABASE_CONFIG \
  -l "hostname=c*" --mem 4G --gpu 1 $EXP_DIR/train.log \
  python local/pyannote/train_ovl_finetune.py $DATASET $EXP_DIR

exit 0