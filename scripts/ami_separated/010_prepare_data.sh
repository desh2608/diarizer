#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/corpora5/amicorpus
SEPARATED_DIR=data/ami_separated_oracle/wav # for oracle
DATA_DIR=data/ami_separated_oracle
EXP_DIR=exp/ami_separated_oracle

mkdir -p exp

if [ $stage -le 0 ]; then
  echo "Preparing separated AMI data..."
  python local/prepare_ami_separated.py --oracle --data-dir $CORPUS_DIR --output-dir $DATA_DIR --separated-dir $SEPARATED_DIR
fi

exit 0
