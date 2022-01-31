#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/corpora5/amicorpus
DATA_DIR=data/ami
EXP_DIR=exp/ami

mkdir -p exp

if [ ! -d AMI-diarization-setup ]; then
  echo "Cloning into AMI-diarization-setup repo (needed for reference RTTMs)."
  git clone https://github.com/BUTSpeechFIT/AMI-diarization-setup.git
fi

if [ $stage -le 0 ]; then
  python local/prepare_ami.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

exit 0
