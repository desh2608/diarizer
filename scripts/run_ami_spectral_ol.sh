#!/usr/bin/env bash
stage=0

. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/export/corpora5/amicorpus
DATA_DIR=data/ami
EXP_DIR=exp/ami

mkdir -p exp

# Overlap detector Hyperparameters (tuned on dev)
onset=0.3
offset=0.7
min_duration_on=0.4
min_duration_off=0.5

if [ ! -d $DATA_DIR ]; then
  echo "First run run_ami_vbx.sh stages 0-4 and then run this script."
  exit 1
fi

if [ $stage -le 5 ]; then
  echo "Onset: $onset Offset: $offset Min_duration_on: $min_duration_on Min_duration_off: $min_duration_off"
  for part in dev test; do
    echo "Running overlap detection on $part set..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/queue.pl -l "hostname=c*" --mem 2G \
        $EXP_DIR/${part}/log/ovl/ovl_${filename}.log \
        python diarizer/overlap/pyannote_overlap.py \
          --model diarizer/models/pyannote/ami_epoch0_step1791.ckpt \
          --in-dir $DATA_DIR/${part}/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/${part}/ovl \
          --onset ${onset} --offset ${offset} \
          --min-duration-on ${min_duration_on} \
          --min-duration-off ${min_duration_off} & 
    done
    wait
    )
    rm exp/list_*.txt
  done
fi

if [ $stage -le 6 ]; then
  for part in dev test; do
    echo "Evaluating ${part} overlap detector output"
    cat $DATA_DIR/${part}/rttm/* | local/get_overlap_segments.py | grep overlap > exp/ref.rttm
    cat $EXP_DIR/${part}/ovl/*.rttm > exp/hyp.rttm
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm |\
      awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/)'
  done
fi

if [ $stage -le 7 ]; then
  for split in dev test; do
    echo "Running spectral clustering on $split..."
    (
    for audio in $(ls $DATA_DIR/${split}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      utils/queue.pl --mem 2G -l hostname="!b03*" $EXP_DIR/$split/log/spectral_ovl/sc_${filename}.log \
        python diarizer/spectral/sclust.py \
          --out-rttm-dir $EXP_DIR/$split/spectral_ovl \
          --xvec-ark-file $EXP_DIR/$split/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$split/xvec/${filename}.seg \
          --overlap-rttm $EXP_DIR/$split/ovl/${filename}.rttm \
          --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 &
    done
    wait
    )
  done
fi

if [ $stage -le 8 ]; then
  for split in dev test; do
    echo "Evaluating $split"
    cat $DATA_DIR/$split/rttm_but/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$split/spectral_ovl/*.rttm > exp/hyp.rttm
    LC_ALL= spyder --per-file exp/ref.rttm exp/hyp.rttm
  done
fi

exit 0
