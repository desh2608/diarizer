#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/fisher
EXP_DIR=exp/fisher

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
  echo "Preparing data in Kaldi format"
  for part in dev test; do
    kaldi_data_dir=data/fisher_${part}_kaldi
    mkdir $kaldi_data_dir
    for f in $DATA_DIR/$part/audios/*.wav; do
      filename=$(basename $f .wav)
      fullpath=$(realpath $f)
      echo "$filename $fullpath"
    done > $kaldi_data_dir/wav.scp
    awk '{print $1, $1}' $kaldi_data_dir/wav.scp > $kaldi_data_dir/utt2spk
    utils/fix_data_dir.sh $kaldi_data_dir
  done
fi

if [ $stage -le 2 ]; then
  for part in dev test; do
    kaldi_data_dir=data/fisher_${part}_kaldi
    echo "Running Aspire VAD on ${part}..."
    steps/segmentation/detect_speech_activity.sh --cmd "queue.pl --mem 2G -l hostname=!b02*" \
      --nj 40 --stage 0 --convert-data-dir-to-whole false --segment-padding 0.6 --acwt 0.6 \
      --graph_opts "--min-silence-duration=0.05 --min-speech-duration=0.02 --max-speech-duration=20.0" \
      $kaldi_data_dir exp/segmentation_1a/tdnn_stats_asr_sad_1a \
      mfcc $EXP_DIR/$part/kaldi_seg $kaldi_data_dir
  done
fi

if [ $stage -le 3 ]; then
  echo "Convert Kaldi output to lab file"
  for part in dev test; do
    mkdir -p $EXP_DIR/$part/vad_kaldi
    kaldi_data_dir=data/fisher_${part}_kaldi
    files=$(cut -d' ' -f1 $kaldi_data_dir/wav.scp)
    for f in $files; do
      grep $f ${kaldi_data_dir}_seg/segments | \
        awk '{print $3, $4, "sp"}' > $EXP_DIR/$part/vad_kaldi/$f.lab
    done
  done
fi

if [ $stage -le 4 ]; then
  for part in dev test; do
    echo "Evaluating ${part} VAD output"
    cat $DATA_DIR/${part}/rttm/* > exp/ref.rttm
    > exp/hyp.rttm
    for x in $EXP_DIR/${part}/vad_kaldi/*; do
      session=$(basename $x .lab)
      awk -v SESSION=${session} \
        '{print "SPEAKER", SESSION, "1", $1, $2-$1, "<NA> <NA> sp <NA> <NA>"}' $x >> exp/hyp.rttm
    done
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm -c 0.25 |\
      awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'
  done
fi

exit 0
