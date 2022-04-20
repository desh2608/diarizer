#!/usr/bin/env bash
stage=0

aligned=false

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/libricss_separated_oracle
EXP_DIR=exp/libricss_separated_oracle

mkdir -p exp

aligned_affix=""
if [[ $aligned == true ]]; then
  aligned_affix="_aligned"
fi

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Extracting x-vectors for $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt

      mkdir -p $EXP_DIR/${part}/xvec${aligned_affix}

      # run feature and x-vectors extraction
      utils/queue-ackgpu.pl -l "hostname=c0*\&!c07*" --gpu 1 --mem 2G \
        $EXP_DIR/${part}/log/xvec${aligned_affix}/xvec_${filename}.log \
        python diarizer/xvector/predict.py \
            --gpus true \
            --in-file-list exp/list_${filename}.txt \
            --in-lab-dir $EXP_DIR/${part}/vad${aligned_affix} \
            --in-wav-dir $DATA_DIR/${part}/audios \
            --out-ark-fn $EXP_DIR/${part}/xvec${aligned_affix}/${filename}.ark \
            --out-seg-fn $EXP_DIR/${part}/xvec${aligned_affix}/${filename}.seg \
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

exit 0
