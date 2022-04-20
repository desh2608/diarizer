#!/usr/bin/env bash
stage=0

# Overlap detector Hyperparameters (tuned on dev)
onset=0.3
offset=0.7
min_duration_on=0.4
min_duration_off=0.5

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/ami
EXP_DIR=exp/ami

mkdir -p exp

if [ $stage -le 0 ]; then
  if [ -f diarizer/models/pyannote/ami_epoch0_step1791.ckpt ]; then
    echo "Found existing AMI pyannote segmentation model, skipping training..."
  else
    echo "Fine tuning pyannote segmentation model on AMI SDM..."
    local/pyannote/train_seg_finetune.sh --DATASET AMI --EXP_DIR exp/pyannote/ami
    cp exp/pyannote/ami/lightning_logs/version_0/checkpoints/epoch=0-step=1791.ckpt diarizer/models/pyannote/ami_epoch0_step1791.ckpt
  fi
fi

if [ $stage -le 1 ]; then
  echo "Onset: $onset Offset: $offset Min_duration_on: $min_duration_on Min_duration_off: $min_duration_off"
  for part in dev test; do
    echo "Running overlap detection on $part set..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      $train_cmd $EXP_DIR/${part}/log/ovl/ovl_${filename}.log \
        python diarizer/overlap/pyannote_overlap.py \
          --model diarizer/models/pyannote/ami_epoch0_step1791.ckpt \
          --in-dir $DATA_DIR/${part}/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/${part}/ovl \
          --onset ${onset} --offset ${offset} \
          --min-duration-on ${min_duration_on} \
          --min-duration-off ${min_duration_off} & 
    done
    wait
    )
    rm exp/list_*.txt
  done
fi

if [ $stage -le 2 ]; then
  for part in dev test; do
    echo "Evaluating ${part} overlap detector output"
    cat $DATA_DIR/${part}/rttm/* | local/get_overlap_segments.py | grep overlap > exp/ref.rttm
    cat $EXP_DIR/${part}/ovl/*.rttm > exp/hyp.rttm
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm |\
      awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/)'
  done
fi

exit 0
