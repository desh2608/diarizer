#!/usr/bin/env bash
stage=0

# VAD Hyperparameters (tuned on dev)
onset=0.6
offset=0.5
min_duration_on=0.5
min_duration_off=0.3

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/callhome_2spk
EXP_DIR=exp/callhome_2spk

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
  for part in dev; do
    echo "Running pyannote VAD on ${part}..."
    (
    for audio in $(ls $DATA_DIR/$part/audios_16k/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/queue.pl -l "hostname=c*" --mem 2G \
        $EXP_DIR/${part}/log/vad/vad_${filename}.log \
        python diarizer/vad/pyannote_vad.py \
          --model diarizer/models/pyannote/callhome_epoch4_step974.ckpt \
          --in-dir $DATA_DIR/$part/audios_16k \
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
  for part in dev; do
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
