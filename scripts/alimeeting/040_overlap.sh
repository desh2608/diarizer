#!/usr/bin/env bash
stage=0

# Overlap detector Hyperparameters (tuned on dev)
onset=0.5
offset=0.6
min_duration_on=0.4
min_duration_off=0.5

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/alimeeting
EXP_DIR=exp/alimeeting

mkdir -p exp

if [ $stage -le 0 ]; then
  if [ -f diarizer/models/pyannote/alimeeting_epoch0_step2257.ckpt ]; then
    echo "Found existing AliMeeting pyannote segmentation model, skipping training..."
  else
    mkdir -p exp/pyannote/alimeeting/lists
    cp data/alimeeting/{train,dev}/rttm/* exp/pyannote/alimeeting/lists/
    for f in $DATA_DIR/{train,dev}/audios/*; do
      filename=$(basename $f .wav)
      duration=$(soxi -D $f)
      echo "$filename 1 0.00 $duration" > exp/pyannote/alimeeting/lists/${filename}.uem
    done
    ls -1 data/alimeeting/train/audios/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/alimeeting/lists/train.meetings.txt
    ls -1 data/alimeeting/dev/audios/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/alimeeting/lists/dev.meetings.txt
    echo "Fine tuning pyannote segmentation model on AliMeeting..."
    local/pyannote/train_seg_finetune.sh --DATASET AliMeeting --EXP_DIR exp/pyannote/alimeeting
    cp exp/pyannote/alimeeting/lightning_logs/version_0/checkpoints/epoch=0-step=2257.ckpt diarizer/models/pyannote/alimeeting_epoch0_step2257.ckpt
  fi
fi

if [ $stage -le 1 ]; then
  for part in eval test; do
    echo "Running overlap detection on $part set..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      $train_cmd $EXP_DIR/${part}/log/ovl/ovl_${filename}.log \
        python diarizer/overlap/pyannote_overlap.py \
          --model diarizer/models/pyannote/alimeeting_epoch0_step2492.ckpt \
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
  echo "Onset: $onset Offset: $offset Min_duration_on: $min_duration_on Min_duration_off: $min_duration_off"
  for part in eval test; do
    echo "Evaluating ${part} overlap detector output"
    cat $DATA_DIR/${part}/rttm/* | local/get_overlap_segments.py | grep overlap > exp/ref.rttm
    cat $EXP_DIR/${part}/ovl/*.rttm > exp/hyp.rttm
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm |\
      awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/)'
  done
fi

exit 0
