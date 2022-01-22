#!/usr/bin/env bash
# First run stages 0-2 in run_libricss_vbx.sh and then run this script.
stage=3

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/LibriCSS
DATA_DIR=data/libricss
EXP_DIR=exp/libricss
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p exp

# Hyperparameters (tuned on session0)
Fa=0.1
Fb=5
loopP=0.9

if [ ! -d $DATA_DIR ]; then
  echo "First run run_libricss_vbx.sh stages 0-2 and then run this script."
  exit 1
fi

if [ $stage -le 3 ]; then
  echo "Running overlap detection..."
  (
  for audio in $(ls $DATA_DIR/audios/*.wav | xargs -n 1 basename)
  do
    filename=$(echo "${audio}" | cut -f 1 -d '.')
    echo ${filename} > exp/list_${filename}.txt

    utils/queue.pl -l "hostname=c*" --mem 2G \
      $EXP_DIR/log/ovl/ovl_${filename}.log \
      python diarizer/overlap/pyannote_overlap.py \
        --in-dir $DATA_DIR/audios \
        --file-list exp/list_${filename}.txt \
        --out-dir $EXP_DIR/ovl \
        --model ovl_dihard &
    
    sleep 10
  done
  wait
  )
  rm exp/list_*.txt
fi

if [ $stage -le 4 ]; then
  echo "Running spectral clustering..."
  (
  for audio in $(ls $DATA_DIR/audios/*.wav | xargs -n 1 basename)
  do
    filename=$(echo "${audio}" | cut -f 1 -d '.')

    # run variational bayes on top of x-vectors
    utils/queue.pl --mem 2G $EXP_DIR/log/spectral_ovl/spectral_${filename}.log \
      python diarizer/spectral/sclust.py \
          --out-rttm-dir $EXP_DIR/out_spectral_ovl \
          --xvec-ark-file $EXP_DIR/xvec/${filename}.ark \
          --segments-file $EXP_DIR/xvec/${filename}.seg \
          --overlap-rttm $EXP_DIR/ovl/${filename}.rttm \
          --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 &
  done
  wait
  )
fi

if [ $stage -le 5 ]; then
  # Combine all RTTM files and score
  cat $DATA_DIR/rttm/*.rttm > $EXP_DIR/ref.rttm
  cat $EXP_DIR/out_spectral_ovl/*.rttm > $EXP_DIR/hyp.rttm
  LC_ALL= spyder $EXP_DIR/ref.rttm $EXP_DIR/hyp.rttm
fi

exit 0
