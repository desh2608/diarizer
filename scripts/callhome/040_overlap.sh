#!/usr/bin/env bash
stage=0

# Overlap detector Hyperparameters (tuned on dev)
onset=0.4
offset=0.3
min_duration_on=0.4
min_duration_off=0.4

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/callhome
EXP_DIR=exp/callhome

mkdir -p exp

if [ $stage -le 0 ]; then
  if [ -f diarizer/models/pyannote/callhome_epoch4_step974.ckpt ]; then
    echo "Found existing callhome pyannote segmentation model, skipping training..."
  else
    mkdir -p exp/pyannote/callhome/lists
    cp data/callhome/{dev,test}/rttm/* exp/pyannote/callhome/lists/
    for f in $DATA_DIR/{dev,test}/audios_16k/*; do
      filename=$(basename $f .wav)
      duration=$(soxi -D $f)
      echo "$filename 1 0.00 $duration" > exp/pyannote/callhome/lists/${filename}.uem
    done
    ls -1 data/callhome/dev/audios_16k/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/callhome/lists/dev.meetings.txt
    ls -1 data/callhome/test/audios_16k/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/callhome/lists/test.meetings.txt
    echo "Fine tuning pyannote segmentation model on Callhome..."
    local/pyannote/train_seg_finetune.sh --DATASET Callhome --EXP_DIR exp/pyannote/callhome
    cp exp/pyannote/callhome/lightning_logs/version_0/checkpoints/epoch=4-step=974.ckpt diarizer/models/pyannote/callhome_epoch4_step974.ckpt
  fi
fi

if [ $stage -le 1 ]; then
  echo "Onset: $onset Offset: $offset Min_duration_on: $min_duration_on Min_duration_off: $min_duration_off"
  for part in dev test; do
    echo "Running overlap detection on $part set..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios_16k/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/queue.pl -l "hostname=c*" --mem 2G \
        $EXP_DIR/${part}/log/ovl/ovl_${filename}.log \
        python diarizer/overlap/pyannote_overlap.py \
          --model diarizer/models/pyannote/callhome_epoch0_step194.ckpt \
          --in-dir $DATA_DIR/${part}/audios_16k \
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
