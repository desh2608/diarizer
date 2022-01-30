#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/AISHELL-4
DATA_DIR=data/aishell4
EXP_DIR=exp/aishell4

mkdir -p exp

# VAD Hyperparameters (tuned on dev)
onset=0.3
offset=0.1
min_duration_on=0.15
min_duration_off=0.8

# Hyperparameters (tuned on dev)
Fa=0.5
Fb=40
loopP=0.9

if [ $stage -le 0 ]; then
  echo "Preparing AISHELL-4 data..."
  utils/queue.pl -l "hostname=c*" --mem 2G $EXP_DIR/log/prepare.log \
    python local/prepare_aishell4.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

if [ $stage -le 1 ]; then
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

if [ $stage -le 2 ]; then
  for part in dev test; do
    echo "Running pyannote VAD on ${part}..."
    (
    for audio in $(ls $DATA_DIR/$part/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/queue.pl -l "hostname=c*" --mem 2G \
        $EXP_DIR/${part}/log/vad/vad_${filename}.log \
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

if [ $stage -le 3 ]; then
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

if [ $stage -le 4 ]; then
  for part in dev test; do
    echo "Extracting x-vectors for ${part}..."
    mkdir -p $EXP_DIR/$part/xvec
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/retry.pl utils/queue-freegpu.pl -l "hostname=c*" --gpu 1 --mem 2G \
        $EXP_DIR/${part}/log/xvec/xvec_${filename}.log \
        python diarizer/xvector/predict.py \
          --gpus true \
          --in-file-list exp/list_${filename}.txt \
          --in-lab-dir $EXP_DIR/${part}/vad \
          --in-wav-dir $DATA_DIR/${part}/audios \
          --out-ark-fn $EXP_DIR/${part}/xvec/${filename}.ark \
          --out-seg-fn $EXP_DIR/${part}/xvec/${filename}.seg \
          --model ResNet101 \
          --weights diarizer/models/ResNet101_16kHz/nnet/raw_81.pth \
          --backend pytorch &
      
      sleep 10
    done
    wait
    )
    rm exp/list_*
  done
fi

if [ $stage -le 5 ]; then
  for part in test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      utils/queue.pl --mem 2G -l hostname="b1*" $EXP_DIR/$part/log/vbx/vb_${filename}.log \
        python diarizer/vbx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir $EXP_DIR/$part/vbx \
          --xvec-ark-file $EXP_DIR/$part/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$part/xvec/${filename}.seg \
          --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 \
          --plda-file diarizer/models/ResNet101_16kHz/plda \
          --threshold -0.015 \
          --init-smoothing 7.0 \
          --lda-dim 128 \
          --Fa $Fa \
          --Fb $Fb \
          --loopP $loopP &
    done
    wait
    )
  done
fi

if [ $stage -le 6 ]; then
  for part in dev test; do
    echo "Evaluating $part"
    cat $DATA_DIR/$part/rttm/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$part/vbx/*.rttm > exp/hyp.rttm
    LC_ALL= spyder --per-file exp/ref.rttm exp/hyp.rttm
  done
fi

exit 0
