#!/usr/bin/env bash
# First run stages 0-3 in run_libricss_spectral.sh and then run this script.
stage=4

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/c01/corpora6/LibriCSS
DATA_DIR=data/libricss
EXP_DIR=exp/libricss
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p exp

# Overlap detector Hyperparameters (tuned on session0)
onset=0.3
offset=0.7
min_duration_on=0.4
min_duration_off=0.5

# VBx Hyperparameters (tuned on session0)
Fa=0.1
Fb=5
loopP=0.9

if [ ! -d $DATA_DIR ]; then
  echo "First run run_libricss_vbx.sh stages 0-3 and then run this script."
  exit 1
fi

if [ $stage -le 4 ]; then
  for part in dev test; do
    echo "Running pyannote Overlap Detection on ${part}"
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/queue.pl -l "hostname=c*" --mem 2G \
        $EXP_DIR/${part}/log/ovl/ovl_${filename}.log \
        python diarizer/overlap/pyannote_overlap.py \
          --in-dir $DATA_DIR/${part}/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/${part}/ovl \
          --onset ${onset} --offset ${offset} \
          --min-duration-on ${min_duration_on} \
          --min-duration-off ${min_duration_off} & 
    done
    wait
    )
    rm -rf exp/list_*
  done
fi

if [ $stage -le 5 ]; then
  for part in dev test; do
    echo "Evaluating ${part} overlap detector output"
    cat $DATA_DIR/${part}/rttm/* | local/get_overlap_segments.py | grep overlap > exp/ref.rttm
    cat $EXP_DIR/${part}/ovl/*.rttm > exp/hyp.rttm
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm |\
      awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/)'
  done
fi

if [ $stage -le 6 ]; then
  for part in dev test; do
    echo "Running spectral clustering on ${part}..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')

      utils/queue.pl --mem 2G $EXP_DIR/log/spectral_ovl/spectral_${filename}.log \
        python diarizer/spectral/sclust.py \
            --out-rttm-dir $EXP_DIR/${part}/spectral_ovl \
            --xvec-ark-file $EXP_DIR/${part}/xvec/${filename}.ark \
            --segments-file $EXP_DIR/${part}/xvec/${filename}.seg \
            --overlap-rttm $EXP_DIR/${part}/ovl/${filename}.rttm \
            --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 &
    done
    wait
    )
  done
fi

if [ $stage -le 7 ]; then
  # Combine all RTTM files and score
  for part in dev test; do
    cat $DATA_DIR/${part}/rttm/*.rttm > $EXP_DIR/ref.rttm
    cat $EXP_DIR/${part}/spectral_ovl/*.rttm > $EXP_DIR/hyp.rttm
    LC_ALL= spyder --per-file $EXP_DIR/ref.rttm $EXP_DIR/hyp.rttm
  done
fi

exit 0
