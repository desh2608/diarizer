#!/usr/bin/env bash
# This script is modified from run_example.sh and uses Lhotse to first prepare the
# data in the required format before running the model.
. ./path.sh

DATA_DIR=/export/c01/corpora6/LibriCSS
EXP_DIR=exp/libricss
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p exp

# Hyperparameters (tuned on session0)
Fa=0.1
Fb=5
loopP=0.9

python local/prepare_libricss.py --data-dir $DATA_DIR --output-dir example/libricss

echo "Running with Fa=$Fa, Fb=$Fb, loopP=$loopP"
(
for audio in $(ls example/libricss/audios/*.wav | xargs -n 1 basename)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list.txt

      # run feature and x-vectors extraction
      utils/queue-freegpu.pl -l "hostname=c*" --gpu 1 --mem 2G \
        exp/libricss/log/xvec_${filename}.log \
        python VBx/predict.py \
            --gpus true \
            --in-file-list exp/list.txt \
            --in-lab-dir example/libricss/vad \
            --in-wav-dir example/libricss/audios \
            --out-ark-fn $EXP_DIR/${filename}.ark \
            --out-seg-fn $EXP_DIR/${filename}.seg \
            --model ResNet101 \
            --weights VBx/models/ResNet101_16kHz/nnet/raw_81.pth \
            --backend pytorch

      # run variational bayes on top of x-vectors
      utils/queue.pl --mem 2G $EXP_DIR/log/vb_${filename}.log \
        python VBx/vbhmm.py \
            --init AHC+VB \
            --out-rttm-dir $EXP_DIR/out \
            --xvec-ark-file $EXP_DIR/${filename}.ark \
            --segments-file $EXP_DIR/${filename}.seg \
            --xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
            --plda-file VBx/models/ResNet101_16kHz/plda \
            --threshold -0.015 \
            --lda-dim 128 \
            --Fa $Fa \
            --Fb $Fb \
            --loopP $loopP &
done
wait
)
# Combine all RTTM files and score
cat example/libricss/rttm/*.rttm > exp/ref.rttm
cat exp/libricss/out/*.rttm > exp/hyp.rttm
python dscore/score.py -r exp/ref.rttm -s exp/hyp.rttm

