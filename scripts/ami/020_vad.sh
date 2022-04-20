#!/usr/bin/env bash
stage=0

# VAD Hyperparameters (tuned on dev)
onset=0.5
offset=0.6
min_duration_on=0.5
min_duration_off=0.1

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
  for part in dev test; do
    echo "Running pyannote VAD on ${part}..."
    (
    for audio in $(ls $DATA_DIR/$part/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      $train_cmd $EXP_DIR/${part}/log/vad/vad_${filename}.log \
        python diarizer/vad/pyannote_vad.py \
          --model diarizer/models/pyannote/ami_epoch0_step1791.ckpt \
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
  for part in dev test; do
    echo "Evaluating ${part} VAD output"
    cat $DATA_DIR/${part}/rttm_but/* > exp/ref.rttm
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
