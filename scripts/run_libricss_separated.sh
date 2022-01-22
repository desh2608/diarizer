#!/usr/bin/env bash
# This script is modified from run_example.sh and uses Lhotse to first prepare the
# data in the required format before running the model.
. ./path.sh

DATA_DIR=/export/c01/corpora6/LibriCSS
EXP_DIR=exp/libricss_separated
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p exp

# Hyperparameters (tuned on session0)
Fa=0.1
Fb=5
loopP=0.9

# python local/prepare_libricss.py --data-dir $DATA_DIR \
#   --separated-dir /export/c03/zhuc/css/connected_continuous_separation/ \
#   --output-dir example/libricss_separated

# Run VAD on separated streams
# (
# for audio in $(ls example/libricss_separated/audios/*.wav | xargs -n 1 basename)
# do
#   filename=$(echo "${audio}" | cut -f 1 -d '.')
#   echo ${filename} > exp/list_${filename}.txt
#   utils/queue.pl -l "hostname=c*" --mem 2G \
#   $EXP_DIR/log/vad_${filename}.log \
#   python diarizer/vad/pyannote_vad.py \
#     --in-dir example/libricss_separated/audios \
#     --file-list exp/list_${filename}.txt \
#     --out-dir $EXP_DIR/vad_separated \
#     --model sad_dihard &
#   sleep 10
# done
# wait
# )
# rm -rf exp/list_*

echo "Running with Fa=$Fa, Fb=$Fb, loopP=$loopP"
(
for audio in $(ls example/libricss_separated/audios/*.wav | xargs -n 1 basename)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt

      # run feature and x-vectors extraction
      # utils/queue.pl -l "hostname=c*" --mem 2G \
      #   exp/libricss_separated/log/xvec_${filename}.log \
      #   python diarizer/xvector/predict.py \
      #       --in-file-list exp/list_${filename}.txt \
      #       --in-lab-dir exp/libricss_separated/vad_separated \
      #       --in-wav-dir example/libricss_separated/audios \
      #       --out-ark-fn $EXP_DIR/${filename}.ark \
      #       --out-seg-fn $EXP_DIR/${filename}.seg \
      #       --model ResNet101 \
      #       --weights diarizer/models/ResNet101_16kHz/nnet/raw_81.pth \
      #       --backend pytorch &

      # run variational bayes on top of x-vectors
      # utils/queue.pl --mem 2G $EXP_DIR/log/vb_${filename}.log \
      #   python diarizer/vbx/vbhmm.py \
      #       --init AHC+VB \
      #       --out-rttm-dir $EXP_DIR/out \
      #       --xvec-ark-file $EXP_DIR/${filename}.ark \
      #       --segments-file $EXP_DIR/${filename}.seg \
      #       --xvec-transform diarizer/models/ResNet101_16kHz/transform.h5 \
      #       --plda-file diarizer/models/ResNet101_16kHz/plda \
      #       --threshold -0.015 \
      #       --lda-dim 128 \
      #       --Fa $Fa \
      #       --Fb $Fb \
      #       --loopP $loopP &
done
wait
)
# Combine all RTTM files and score
# cat example/libricss/rttm/*.rttm > exp/ref.rttm
# cat exp/libricss/out/*.rttm > exp/hyp.rttm
# python dscore/score.py -r exp/ref.rttm -s exp/hyp.rttm

