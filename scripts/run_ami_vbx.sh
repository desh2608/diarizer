#!/usr/bin/env bash
# This script requires the BUT AMI setup for data preparation. Please run:
# git clone https://github.com/BUTSpeechFIT/AMI-diarization-setup.git
# before running this script.
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/corpora5/amicorpus
DATA_DIR=data/ami
EXP_DIR=exp/ami

mkdir -p exp

# Hyperparameters (from original repo)
Fa=0.4
Fb=64
loopP=0.65

if [ ! -d AMI-diarization-setup ]; then
  echo "Please clone AMI-diarization-setup repo before running this script."
  exit 1
fi

if [ $stage -le 0 ]; then
  python local/prepare_ami.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

if [ $stage -le 1 ]; then
  for split in dev test; do
    echo "Running pyannote VAD on ${split}..."
    (
    for audio in $(ls $DATA_DIR/$split/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/queue.pl -l "hostname=c*" --mem 2G \
        $EXP_DIR/${split}/log/vad/vad_${filename}.log \
        python diarizer/vad/pyannote_vad.py \
          --in-dir $DATA_DIR/$split/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/$split/vad \
          --model sad_dihard &
      
      sleep 10
    done
    wait
    )
    rm exp/list_*
  done
fi

if [ $stage -le 2 ]; then
  for split in dev test; do
    echo "Extracting x-vectors for ${split}..."
    mkdir -p $EXP_DIR/$split/xvec
    (
    for audio in $(ls $DATA_DIR/${split}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/retry.pl utils/queue-freegpu.pl -l "hostname=c*" --gpu 1 --mem 2G \
        $EXP_DIR/${split}/log/xvec/xvec_${filename}.log \
        python diarizer/xvector/predict.py \
          --gpus true \
          --in-file-list exp/list_${filename}.txt \
          --in-lab-dir $EXP_DIR/${split}/vad \
          --in-wav-dir $DATA_DIR/${split}/audios \
          --out-ark-fn $EXP_DIR/${split}/xvec/${filename}.ark \
          --out-seg-fn $EXP_DIR/${split}/xvec/${filename}.seg \
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

if [ $stage -le 3 ]; then
  for split in dev test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $split..."
    (
    for audio in $(ls $DATA_DIR/${split}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      utils/queue.pl --mem 2G $EXP_DIR/$split/log/vbx/vb_${filename}.log \
        python diarizer/vbx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir $EXP_DIR/$split/out \
          --xvec-ark-file $EXP_DIR/$split/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$split/xvec/${filename}.seg \
          --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 \
          --plda-file diarizer/models/ResNet101_16kHz/plda \
          --threshold -0.015 \
          --init-smoothing 7.0 \
          --lda-dim 128 \
          --Fa $Fa \
          --Fb $Fb \
          --loopP $loopP &
      sleep 10
    done
    wait
    )
  done
fi

if [ $stage -le 4 ]; then
  for split in dev; do
    echo "Evaluating $split"
    cat $DATA_DIR/$split/rttm/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$split/out/*.rttm > exp/hyp.rttm
    LC_ALL= spyder --per-file exp/ref.rttm exp/hyp.rttm
  done
fi

exit 0
