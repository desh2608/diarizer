#!/usr/bin/env bash
stage=0
nj=10

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/fisher
EXP_DIR=exp/fisher

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Splitting $part into $nj parts"
    > $DATA_DIR/files_$part.txt
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} >> $DATA_DIR/files_$part.txt
    done
    split --number l/${nj} --numeric-suffixes=1 --additional-suffix=".txt" \
      $DATA_DIR/files_$part.txt exp/list_$part.
    # Remove 0 padding from filenames
    rename 's/\.0/\./' exp/list_$part.*.txt
  done
fi

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Extracting x-vectors for ${part}..."
    mkdir -p $EXP_DIR/$part/xvec
    (
    utils/queue-freegpu.pl -l "hostname=c*\&!c21*" --gpu 1 --mem 2G --max-jobs-run 4 \
      JOB=1:$nj $EXP_DIR/${part}/log/xvec/extract.JOB.log \
      python diarizer/xvector/predict_batch.py \
        --gpus true \
        --in-file-list exp/list_${part}.JOB.txt \
        --in-lab-dir $EXP_DIR/${part}/vad_kaldi \
        --in-wav-dir $DATA_DIR/${part}/audios \
        --out-ark-dir $EXP_DIR/${part}/xvec \
        --out-seg-dir $EXP_DIR/${part}/xvec \
        --model ResNet101 \
        --weights diarizer/models/ResNet101_8kHz/nnet/raw_195.pth \
        --backend pytorch &
    wait
    )
  done
fi

exit 0
