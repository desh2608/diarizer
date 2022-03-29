#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR="/export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13"
DATA_DIR=data/fisher_train
EXP_DIR=exp/fisher_train

mkdir -p exp

if [ $stage -le 0 ]; then
  python local/prepare_fisher_train.py --corpus-dirs $CORPUS_DIR --data-dir $DATA_DIR
fi

if [ $stage -le 1 ]; then
  # Prepare utt2spk and spk2utt files
  awk '{print $1, $1}' $DATA_DIR/wav.scp > $DATA_DIR/utt2spk
  utils/utt2spk_to_spk2utt.pl $DATA_DIR/utt2spk > $DATA_DIR/spk2utt
  utils/fix_data_dir.sh $DATA_DIR
fi

exit 0
