#!/usr/bin/env bash
stage=0

# Overlap detector Hyperparameters (tuned on session0)
onset=0.3
offset=0.7
min_duration_on=0.4
min_duration_off=0.5

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/libricss
EXP_DIR=exp/libricss

mkdir -p exp

if [ $stage -le 4 ]; then
  for part in dev test; do
    echo "Running pyannote Overlap Detection on ${part}"
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      $train_cmd $EXP_DIR/${part}/log/ovl/ovl_${filename}.log \
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

exit 0
