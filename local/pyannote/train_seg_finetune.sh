#!/bin/bash

DATASET=AMI
EXP_DIR=exp/pyannote/ami

. ./path.sh
. ./utils/parse_options.sh

export PYANNOTE_DATABASE_CONFIG=local/pyannote/database.yml

mkdir -p $EXP_DIR

utils/queue-ackgpu.pl -v PYANNOTE_DATABASE_CONFIG \
  -l "hostname=c*" --mem 32G --gpu 1 $EXP_DIR/train.log \
  python local/pyannote/train_seg_finetune.py $DATASET $EXP_DIR

exit 0