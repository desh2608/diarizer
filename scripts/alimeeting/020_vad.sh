#!/usr/bin/env bash
stage=0

# VAD Hyperparameters (tuned on eval)
onset=0.6
offset=0.5
min_duration_on=0.4
min_duration_off=0.15

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/alimeeting
EXP_DIR=exp/alimeeting

mkdir -p exp

if [ $stage -le 0 ]; then
  if [ -f diarizer/models/pyannote/alimeeting_epoch0_step2492.ckpt ]; then
    echo "Found existing AliMeeting pyannote segmentation model, skipping training..."
  else
    mkdir -p exp/pyannote/alimeeting/lists
    cp data/alimeeting/{train,eval}/rttm/* exp/pyannote/alimeeting/lists/
    for f in $DATA_DIR/{train,eval}/audios/*; do
      filename=$(basename $f .wav)
      duration=$(soxi -D $f)
      echo "$filename 1 0.00 $duration" > exp/pyannote/alimeeting/lists/${filename}.uem
    done
    ls -1 data/alimeeting/train/audios/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/alimeeting/lists/train.meetings.txt
    ls -1 data/alimeeting/eval/audios/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/alimeeting/lists/eval.meetings.txt
    echo "Fine tuning pyannote segmentation model on AliMeeting..."
    local/pyannote/train_seg_finetune.sh --DATASET AliMeeting --EXP_DIR exp/pyannote/alimeeting
    cp exp/pyannote/alimeeting/lightning_logs/version_0/checkpoints/epoch=0-step=2492.ckpt diarizer/models/pyannote/alimeeting_epoch0_step2492.ckpt
  fi
fi

if [ $stage -le 1 ]; then
  for part in eval test; do
    echo "Running pyannote VAD on ${part}..."
    (
    for audio in $(ls $DATA_DIR/$part/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      $train_cmd $EXP_DIR/${part}/log/vad/vad_${filename}.log \
        python diarizer/vad/pyannote_vad.py \
          --model diarizer/models/pyannote/alimeeting_epoch0_step2492.ckpt \
          --in-dir $DATA_DIR/$part/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/$part/vad \
          --onset ${onset} --offset ${offset} \
          --min-duration-on ${min_duration_on} \
          --min-duration-off ${min_duration_off} & 
      
    done
    wait
    )
    rm exp/list_*
  done
fi

if [ $stage -le 2 ]; then
  echo "Onset: $onset Offset: $offset Min-duration-on: $min_duration_on Min-duration-off: $min_duration_off"
  for part in eval test; do
    echo "Evaluating ${part} VAD output"
    cat $DATA_DIR/${part}/rttm/* > exp/ref.rttm
    > exp/hyp.rttm
    for x in $EXP_DIR/${part}/vad/*; do
      session=$(basename $x .lab)
      awk -v SESSION=${session} \
        '{print "SPEAKER", SESSION, "1", $1, $2-$1, "<NA> <NA> sp <NA> <NA>"}' $x >> exp/hyp.rttm
    done
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm -c 0.25 |\
      awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'
  done
fi

exit 0
