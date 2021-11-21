#!/usr/bin/env bash
# This script is modified from run_example.sh and uses Lhotse to first prepare the
# data in the required format before running the model.
overlap=false

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=/export/corpora5/amicorpus
EXP_DIR=exp/ami

mkdir -p exp

# Hyperparameters (from original repo)
Fa=0.4
Fb=64
loopP=0.65

if [ ! -f example/ami/.done ]; then
  python local/prepare_ami.py --data-dir $DATA_DIR --output-dir example/ami
fi

# run feature and x-vectors extraction
for split in dev test; do
  if [ ! -f $EXP_DIR/$split/xvec.done ]; then
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
  touch $EXP_DIR/$split/xvec.done
  fi
done

# Run overlap detection
overlap_affix=""
if [ $overlap == "true" ]; then
  overlap_affix="_ovl"
  for split in dev test; do
    if [ ! -f $EXP_DIR/$split/overlap.done ]; then
    (
    for audio in $(ls example/ami/${split}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      utils/queue.pl -l "hostname=c*" --mem 2G \
      $EXP_DIR/${split}/log/overlap_${filename}.log \
      python VBx/pyannote_overlap.py \
        --in-dir example/ami/${split}/audios \
        --file-list exp/list_${filename}.txt \
        --out-dir $EXP_DIR/${split}/overlap \
        --model ovl_dihard &
      sleep 10
    done
    wait
    )
    touch $EXP_DIR/$split/overlap.done
    fi
  done
fi

# Run VBx diarization
echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP"
for split in dev test; do
  if [ ! -f $EXP_DIR/$split/vbx.done ]; then
  (
  for audio in $(ls example/ami/${split}/audios/*.wav | xargs -n 1 basename)
  do
    filename=$(echo "${audio}" | cut -f 1 -d '.')
    overlap_opts=""
    if [ $overlap == "true" ]; then
      overlap_opts="--overlap-rttm $EXP_DIR/$split/overlap/${filename}.rttm"
    fi
    utils/queue.pl --mem 2G $EXP_DIR/$split/log/vb${overlap_affix}_${filename}.log \
      python VBx/vbhmm.py $overlap_opts \
        --init AHC+VB \
        --out-rttm-dir $EXP_DIR/$split/out${overlap_affix} \
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
  touch $EXP_DIR/$split/vbx.done
  fi
done

# Combine all RTTM files and score
for split in dev test; do
  echo "Evaluating $split"
  cat example/ami/$split/rttm/*.rttm > exp/ref.rttm
  cat exp/ami/$split/out${overlap_affix}/*.rttm > exp/hyp.rttm
  LC_ALL= spyder exp/ref.rttm exp/hyp.rttm
done

# RESULTS (w/o overlap assignment)
# Evaluating dev
# Evaluated 18 recordings on `all` regions. Results:
# ╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
# │ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
# ╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
# │ Overall     │       35495.22 │  19.51% │      0.00% │   5.51% │ 25.02% │
# ╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛
# Evaluating test
# Evaluated 16 recordings on `all` regions. Results:
# ╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
# │ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
# ╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
# │ Overall     │       33952.95 │  19.91% │      0.00% │   8.32% │ 28.23% │
# ╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛

# RESULTS (w/ overlap assignment)
# Evaluating dev
# Evaluated 18 recordings on `all` regions. Results:
# ╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
# │ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
# ╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
# │ Overall     │       35495.22 │  10.55% │      3.47% │   8.50% │ 22.53% │
# ╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛
# Evaluating test
# Evaluated 16 recordings on `all` regions. Results:
# ╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
# │ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
# ╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
# │ Overall     │       33952.95 │  11.69% │      3.61% │  12.22% │ 27.51% │
# ╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛