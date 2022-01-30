#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/corpora5/amicorpus
DATA_DIR=data/ami
EXP_DIR=exp/ami

mkdir -p exp

# VAD Hyperparameters (tuned on dev)
onset=0.5
offset=0.6
min_duration_on=0.5
min_duration_off=0.1

# Hyperparameters (from original repo)
Fa=0.4
Fb=64
loopP=0.65

if [ ! -d AMI-diarization-setup ]; then
  echo "Cloning into AMI-diarization-setup repo (needed for reference RTTMs)."
  git clone https://github.com/BUTSpeechFIT/AMI-diarization-setup.git
fi

if [ $stage -le 0 ]; then
  python local/prepare_ami.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

if [ $stage -le 1 ]; then
  if [ -f diarizer/models/pyannote/ami_epoch0_step1791.ckpt ]; then
    echo "Found existing AMI pyannote segmentation model, skipping training..."
  else
    echo "Fine tuning pyannote segmentation model on AMI SDM..."
    local/pyannote/train_seg_finetune.sh --DATASET AMI --EXP_DIR exp/pyannote/ami
    cp exp/pyannote/ami/lightning_logs/version_0/checkpoints/epoch=0-step=1791.ckpt diarizer/models/pyannote/ami_epoch0_step1791.ckpt
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
        $EXP_DIR/${part}/log/vad_ft/vad_${filename}.log \
        python diarizer/vad/pyannote_vad.py \
          --model diarizer/models/pyannote/ami_epoch0_step1791.ckpt \
          --in-dir $DATA_DIR/$part/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/$part/vad_ft \
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
  for part in dev test; do
    echo "Evaluating ${part} VAD output"
    cat $DATA_DIR/${part}/rttm_but/* > exp/ref.rttm
    > exp/hyp.rttm
    for x in $EXP_DIR/${part}/vad_ft/*; do
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
          --in-lab-dir $EXP_DIR/${part}/vad_ft \
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
  for part in dev test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      utils/queue.pl --mem 2G -l "hostname=!c13*" $EXP_DIR/$part/log/vbx/vb_${filename}.log \
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
    cat $DATA_DIR/$part/rttm_but/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$part/vbx/*.rttm > exp/hyp.rttm
    LC_ALL= spyder --per-file exp/ref.rttm exp/hyp.rttm
  done
fi

exit 0
