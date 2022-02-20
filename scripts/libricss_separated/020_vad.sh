#!/usr/bin/env bash
stage=0

# VAD Hyperparameters (tuned on session0)
onset=0.7
offset=0.3
min_duration_on=0.3
min_duration_off=0.6

# Specify if aligned channels are required
aligned=false

. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/libricss_separated_oracle
EXP_DIR=exp/libricss_separated_oracle

mkdir -p exp

aligned_opts=""
aligned_affix=""
if [[ $aligned == true ]]; then
  aligned_opts="--align-time 0.24"  # 0.24s is the window shift for x-vector extraction
  aligned_affix="_aligned"
fi

if [ $stage -le 0 ]; then
  for part in dev test; do
    echo "Running pyannote VAD on ${part}"
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      utils/queue.pl -l "hostname=c*" --mem 2G \
        $EXP_DIR/${part}/log/vad${aligned_affix}/vad_${filename}.log \
        python diarizer/vad/pyannote_vad.py $aligned_opts \
          --in-dir $DATA_DIR/${part}/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/${part}/vad${aligned_affix} \
          --onset ${onset} --offset ${offset} \
          --min-duration-on ${min_duration_on} \
          --min-duration-off ${min_duration_off} & 
    done
    wait
    )
    rm -rf exp/list_*
  done
fi

if [ $stage -le 1 ]; then
  for part in dev test; do
    echo "Evaluating ${part} VAD output"
    cat $DATA_DIR/${part}/rttm/* > exp/ref.rttm
    > exp/hyp.rttm
    for x in $EXP_DIR/${part}/vad${aligned_affix}/*; do
      session=$(basename $x .lab)
      # Remove last 2 characters (channel)
      awk -v SESSION=${session} \
        '{t=length(SESSION)}{print "SPEAKER", substr(SESSION,0,t-2), "1", $1, $2-$1, "<NA> <NA> sp <NA> <NA>"}' $x >> exp/hyp.rttm
    done
    ./md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm |\
      awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'
  done
fi

exit 0
