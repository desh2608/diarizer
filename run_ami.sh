#!/usr/bin/env bash
# This script is modified from run_example.sh and uses Lhotse to first prepare the
# data in the required format before running the model.
. ./path.sh

DATA_DIR=/export/corpora5/amicorpus
EXP_DIR=exp/ami

mkdir -p exp

# Hyperparameters (from original repo)
Fa=0.4
Fb=64
loopP=0.65

python local/prepare_ami.py --data-dir $DATA_DIR --output-dir example/ami

# run feature and x-vectors extraction
for split in dev test; do
(
for audio in $(ls example/ami/${split}/audios/*.wav | xargs -n 1 basename)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      utils/retry.pl utils/queue-freegpu.pl -l "hostname=c*" --gpu 1 --mem 2G \
      $EXP_DIR/${split}/log/xvec_${filename}.log \
      python VBx/predict.py \
            --gpus true \
            --in-file-list exp/list_${filename}.txt \
            --in-lab-dir example/ami/${split}/vad \
            --in-wav-dir example/ami/${split}/audios \
            --out-ark-fn $EXP_DIR/${split}/${filename}.ark \
            --out-seg-fn $EXP_DIR/${split}/${filename}.seg \
            --model ResNet101 \
            --weights VBx/models/ResNet101_16kHz/nnet/raw_81.pth \
            --backend pytorch &
      sleep 10
done
wait
)
done

# Run VBx diarization
echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP"
for split in dev test; do
(
for audio in $(ls example/ami/${split}/audios/*.wav | xargs -n 1 basename)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      utils/queue.pl --mem 2G $EXP_DIR/$split/log/vb_${filename}.log \
        python VBx/vbhmm.py \
            --init AHC+VB \
            --out-rttm-dir $EXP_DIR/$split/out \
            --xvec-ark-file $EXP_DIR/$split/${filename}.ark \
            --segments-file $EXP_DIR/$split/${filename}.seg \
            --xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
            --plda-file VBx/models/ResNet101_16kHz/plda \
            --threshold -0.015 \
            --init-smoothing 7.0 \
            --lda-dim 128 \
            --Fa $Fa \
            --Fb $Fb \
            --loopP $loopP &
      sleep 10
done
wait
)
done

# Combine all RTTM files and score
for split in dev test; do
  echo "Evaluating $split"
  cat example/ami/$split/rttm/*.rttm > exp/ref.rttm
  cat exp/ami/$split/out/*.rttm > exp/hyp.rttm
  python dscore/score.py -r exp/ref.rttm -s exp/hyp.rttm
done

# Results:
# dev DER: 25.02
# eval DER: 28.23
