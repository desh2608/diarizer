#!/usr/bin/env bash
stage=0

# VAD Hyperparameters (tuned on dev)
onset=0.3
offset=0.1
min_duration_on=0.15
min_duration_off=0.8

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/aishell4
EXP_DIR=exp/aishell4

mkdir -p exp

if [ $stage -le 0 ]; then
  if [ -f diarizer/models/pyannote/aishell_epoch0_step2150.ckpt ]; then
    echo "Found existing AISHELL-4 pyannote segmentation model, skipping training..."
  else
    mkdir -p exp/pyannote/aishell4/lists
    for f in $DATA_DIR/{train,dev}/audios/*; do
      filename=$(basename $f .wav)
      grep "$filename" local/pyannote/aishell4_train_force_aligned.rttm > exp/pyannote/aishell4/lists/${filename}.rttm
      duration=$(soxi -D $f)
      echo "$filename 1 0.00 $duration" > exp/pyannote/aishell4/lists/${filename}.uem
    done
    ls -1 data/aishell4/train/audios/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/aishell4/lists/train.meetings.txt
    ls -1 data/aishell4/dev/audios/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/aishell4/lists/dev.meetings.txt
    echo "Fine tuning pyannote segmentation model on AISHELL-4..."
    local/pyannote/train_seg_finetune.sh --DATASET AISHELL-4 --EXP_DIR exp/pyannote/aishell4
    cp exp/pyannote/aishell4/lightning_logs/version_0/checkpoints/epoch=0-step=2150.ckpt diarizer/models/pyannote/aishell_epoch0_step2150.ckpt
  fi
fi

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Running pyannote VAD on ${part}..."
    (
    for audio in $(ls $DATA_DIR/$part/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      $train_cmd $EXP_DIR/${part}/log/vad/vad_${filename}.log \
        python diarizer/vad/pyannote_vad.py \
          --model diarizer/models/pyannote/aishell_epoch0_step2150.ckpt \
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
  for part in dev test; do
    echo "Evaluating ${part} VAD output"
    cat $DATA_DIR/${part}/rttm/* > exp/ref.rttm
    > exp/hyp.rttm
    for x in $EXP_DIR/${part}/vad/*; do
      session=$(basename $x .lab)
      awk -v SESSION=${session} \
        '{print "SPEAKER", SESSION, "1", $1, $2-$1, "<NA> <NA> sp <NA> <NA>"}' $x >> exp/hyp.rttm
    done
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm |\
      awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'
  done
fi

exit 0
