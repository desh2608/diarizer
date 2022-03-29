#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/fisher_train
EXP_DIR=exp/fisher_train

mkdir -p exp

if [ -z $KALDI_ROOT ] || [ ! -d $KALDI_ROOT ]; then
  echo "KALDI_ROOT is not set or does not exist! Please check path.sh"
  exit 1
fi

if [ $stage -le 0 ]; then
  echo "Download Aspire SAD model..."
  wget --no-check-certificate https://kaldi-asr.org/models/4/0004_tdnn_stats_asr_sad_1a.tar.gz
  tar -xvf 0004_tdnn_stats_asr_sad_1a.tar.gz
fi

if [ $stage -le 1 ]; then
  echo "Running Aspire VAD..."
  steps/segmentation/detect_speech_activity.sh --cmd "queue.pl --mem 2G -l hostname=!b02*" \
    --nj 40 --stage 0 --convert-data-dir-to-whole false --segment-padding 0.1 --acwt 0.5 \
    --graph_opts "--min-silence-duration=0.05 --min-speech-duration=0.02 --max-speech-duration=10.0" \
    $DATA_DIR exp/segmentation_1a/tdnn_stats_asr_sad_1a \
    mfcc $EXP_DIR/seg $DATA_DIR
fi

if [ $stage -le 2 ]; then
  echo "Convert Kaldi output to lab file"
  mkdir -p $EXP_DIR/vad_kaldi
  files=$(cut -d' ' -f1 $DATA_DIR/wav.scp)
  for f in $files; do
    grep $f ${DATA_DIR}_seg/segments | \
      awk '{print $3, $4, "sp"}' > $EXP_DIR/vad_kaldi/$f.lab
  done
fi

exit 0
