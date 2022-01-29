## Clustering-based Diarization

Python implementations of some clustering-based diarization systems. For details, refer to the [documentation](https://desh2608.github.io/diarizer/).

### Features

* End-to-end recipes (from unsegmented audio to evaluation) for LibriCSS, AMI, and AISHELL-4.
* Using [Lhotse](https://github.com/lhotse-speech/lhotse) for data preparation. 
* Using [Pyannote 2.0](https://github.com/pyannote/pyannote-audio/tree/develop) models for VAD and overlap detection.
* Scripts for fine-tuning Pyannote models on AMI and AISHELL-4 (fine-tuned models also provided).
* VBx and x-vector extraction from [BUT](https://github.com/BUTSpeechFIT/VBx)'s implementation.
* [Kaldi](https://github.com/kaldi-asr/kaldi) implementation of overlap-aware spectral clustering.

### Installation (basic)

```
> git clone https://github.com/desh2608/diarizer.git
> cd diarizer
> pip install -e . 
```

### Usage

End-to-end runnable scripts are provided in the `scripts` directory. You can run them as:

```
> scripts/run_ami_spectral_ol.sh
```

The `--stage` parameter may be passed to restart run from a particular stage.
