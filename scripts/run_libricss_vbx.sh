#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/LibriCSS
DATA_DIR=data/libricss
EXP_DIR=exp/libricss

mkdir -p exp

# VAD Hyperparameters (tuned on session0)
onset=0.3
offset=0.1
min_duration_on=0.3
min_duration_off=0.7

# VBx Hyperparameters (tuned on session0)
Fa=0.1
Fb=5
loopP=0.9

if [ $stage -le 0 ]; then
  echo "Preparing LibriCSS data..."
  python local/prepare_libricss.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Running pyannote VAD on ${part}"
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/queue.pl -l "hostname=c*" --mem 2G \
        $EXP_DIR/${part}/log/vad/vad_${filename}.log \
        python diarizer/vad/pyannote_vad.py \
          --in-dir $DATA_DIR/${part}/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/${part}/vad \
          --onset ${onset} --offset ${offset} \
          --min-duration-on ${min_duration_on} \
          --min-duration-off ${min_duration_off} & 
    done
    wait
    )
    rm -rf exp/list_*
  done
fi

if [ $stage -le 2 ]; then
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

if [ $stage -le 3 ]; then
  for part in dev test; do
    echo "Extracting x-vectors for $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt

      mkdir -p $EXP_DIR/${part}/xvec

      # run feature and x-vectors extraction
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
    rm exp/list_*.txt
  done
fi

if [ $stage -le 4 ]; then
  echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP"
  (
  for audio in $(ls $DATA_DIR/audios/*.wav | xargs -n 1 basename)
  do
    filename=$(echo "${audio}" | cut -f 1 -d '.')

    # run variational bayes on top of x-vectors
    utils/queue.pl --mem 2G $EXP_DIR/log/vbx/vb_${filename}.log \
      python diarizer/vbx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir $EXP_DIR/out \
          --xvec-ark-file $EXP_DIR/xvec/${filename}.ark \
          --segments-file $EXP_DIR/xvec/${filename}.seg \
          --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 \
          --plda-file diarizer/models/ResNet101_16kHz/plda \
          --threshold -0.015 \
          --lda-dim 128 \
          --Fa $Fa \
          --Fb $Fb \
          --loopP $loopP &
  done
  wait
  )
fi

if [ $stage -le 5 ]; then
  # Combine all RTTM files and score
  cat $DATA_DIR/rttm/*.rttm > $EXP_DIR/ref.rttm
  cat $EXP_DIR/out/*.rttm > $EXP_DIR/hyp.rttm
  LC_ALL= spyder $EXP_DIR/ref.rttm $EXP_DIR/hyp.rttm
fi

exit 0
