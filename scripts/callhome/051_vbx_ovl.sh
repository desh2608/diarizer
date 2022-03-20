#!/usr/bin/env bash
stage=0

# Hyperparameters (from original repo)
Fa=0.4
Fb=17
loopP=0.40

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/callhome
EXP_DIR=exp/callhome

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios_8k/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      utils/queue.pl --mem 2G $EXP_DIR/$part/log/vbx_ovl/vb_${filename}.log \
        python diarizer/vbx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir $EXP_DIR/$part/vbx_ovl \
          --xvec-ark-file $EXP_DIR/$part/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$part/xvec/${filename}.seg \
          --overlap-rttm $EXP_DIR/$part/ovl/${filename}.rttm \
          --xvec-transform diarizer/models/ResNet101_8kHz/transform.h5 \
          --plda-file diarizer/models/ResNet101_8kHz/plda \
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

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Evaluating $part"
    cat $DATA_DIR/$part/rttm/*.rttm > exp/ref.rttm
    cat $EXP_DIR/$part/vbx_ovl/*.rttm > exp/hyp.rttm
    LC_ALL= spyder exp/ref.rttm exp/hyp.rttm
  done
fi

if [ $stage -le 2 ]; then
  for part in dev test; do
    # Create 2 speaker list
    for f in data/callhome/$part/rttm/*; do 
      filename=$(basename $f .rttm)
      nspk=$(cut -d' ' -f8 $f | sort -u | wc -l)
      if [ $nspk -eq 2 ]; then 
        echo $filename
      fi
    done > data/callhome/$part/2spk.list
    
    > exp/ref.rttm
    > exp/hyp.rttm
    while IFS= read -r line
    do
      cat $DATA_DIR/$part/rttm/${line}.rttm >> exp/ref.rttm
      cat $EXP_DIR/$part/vbx_ovl/${line}.rttm >> exp/hyp.rttm
    done < $DATA_DIR/$part/2spk.list

    echo "Evaluating on 2-speaker subset (full)"
    LC_ALL= spyder exp/ref.rttm exp/hyp.rttm

    echo "Evaluating on 2-speaker subset (fair)"
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm -c 0.25
  done
fi

exit 0
