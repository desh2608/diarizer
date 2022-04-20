#!/usr/bin/env bash
stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/libricss
EXP_DIR=exp/libricss

mkdir -p exp

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Extracting x-vectors for $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt

      mkdir -p $EXP_DIR/${part}/xvec

      # run feature and x-vectors extraction
      $cuda_cmd $EXP_DIR/${part}/log/xvec/xvec_${filename}.log \
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
    done
    wait
    )
    rm exp/list_*.txt
  done
fi

exit 0
