#!/usr/bin/env bash
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

if [ ! -d $DATA_DIR ]; then
  echo "First run run_ami_vbx.sh stages 0-2 and then run this script."
  exit 1
fi

if [ $stage -le 3 ]; then
  for split in dev test; do
    echo "Running overlap detection on $split set..."
    (
    for audio in $(ls $DATA_DIR/${split}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/queue.pl -l "hostname=c*" --mem 2G \
        $EXP_DIR/${split}/log/ovl_dihard/ovl_${filename}.log \
        python diarizer/overlap/pyannote_overlap.py \
          --in-dir $DATA_DIR/${split}/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/${split}/ovl_dihard \
          --model ovl_dihard &
      
      sleep 10
    done
    wait
    )
    rm exp/list_*.txt
  done
fi

if [ $stage -le 4 ]; then
  for split in dev test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $split..."
    (
    for audio in $(ls $DATA_DIR/${split}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      utils/queue.pl --mem 2G $EXP_DIR/$split/log/vbx_ovl_dihard/vb_${filename}.log \
        python diarizer/vbx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir $EXP_DIR/$split/vbx_ovl_dihard \
          --xvec-ark-file $EXP_DIR/$split/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$split/xvec/${filename}.seg \
          --overlap-rttm $EXP_DIR/$split/ovl_dihard/${filename}.rttm \
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

if [ $stage -le 5 ]; then
  for split in dev test; do
    echo "Evaluating $split"
    cat $DATA_DIR/$split/rttm_but/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$split/vbx_ovl_ami/*.rttm > exp/hyp.rttm
    LC_ALL= spyder --per-file exp/ref.rttm exp/hyp.rttm
  done
fi

exit 0
