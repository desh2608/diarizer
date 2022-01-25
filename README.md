## Clustering-based Diarization

Python implementations of some clustering-based diarization systems.

### Features

* End-to-end recipes (from unsegmented audio to evaluation) for LibriCSS, AMI, and AISHELL-4.
* Using [Lhotse](https://github.com/lhotse-speech/lhotse) for data preparation. 
* Using [Pyannote 2.0](https://github.com/pyannote/pyannote-audio/tree/develop) models for VAD and overlap detection.
* VBx and x-vector extraction from [BUT](https://github.com/BUTSpeechFIT/VBx)'s implementation.
* [Kaldi](https://github.com/kaldi-asr/kaldi) implementation of overlap-aware spectral clustering.

### Installation

We recommend installation within a virtual environment (such as Conda) so that the
package versions are consistent. To create a new Conda environment:

```
> conda create -n diar python=3.8
> conda activate diar
```

```
> git clone https://github.com/desh2608/diarizer.git
> cd diarizer
> pip install -e . 
```

The run scripts additionally use some Kaldi utilities (such as queue.pl or parse_options.sh), 
since we submit multiple jobs (usually 1 job per audio file). You may need to modify these
if you are running in a different environment. Alternatively, if you have Kaldi somewhere, 
you can make a symbolic link to the utils folder as:

```
> KALDI_ROOT=/path/to/kaldi
> ln -s $KALDI_ROOT/egs/wsj/s5/utils .
```

### Usage

End-to-end runnable scripts are provided in the `scripts` directory. You can run them as:

```
> scripts/run_ami_spectral_ol.sh
```

The `--stage` parameter may be passed to restart run from a particular stage.

### Results

1. Voice activity detection (VAD) using Pyannote

| Method   | MS    | FA | Total   |
|----------|-------|----|------|
| LibriCSS | 0.9 | 1.2 | 2.1 | 
| AMI | 6.7 | 3.0 | 9.7 |
| AISHELL-4 |  |    |  |

2. Speaker diarization (using above VAD)

The following is evaluated using the [spyder](https://github.com/desh2608/spyder) package without ignoring overlaps and using a 0.0 collar.

* LibriCSS

| Method   | MS    | FA | Conf. | DER   |
|----------|-------|----|-------|-------|
| VBx | 10.37 | 1.19 | 2.96 | 14.52 |
| VBx + OVL | 3.39 | 2.31 | 5.55 | 11.25 |
| Spectral | 10.37 | 1.19 | 3.37 | 14.93 |
| Spectral + OVL | 3.79 | 2.22 | 5.33 | 11.34 |

* AMI (SDM)

| Method   | MS    | FA | Conf. | DER   |
|----------|-------|----|-------|-------|
| VBx | 21.42 | 3.31 | 5.31 | 30.04 |
| VBx + OVL |  |  |   |  |
| Spectral | 21.42 | 3.31 | 3.83 | 28.56 |
| Spectral + OVL |  |  |   |  |

* AISHELL-4

| Method   | MS    | FA | Conf. | DER   |
|----------|-------|----|-------|-------|
| VBx |  |  |  | |
| VBx + OVL |  |  |   |  |
| Spectral |  |  |   |  |
| Spectral + OVL |  |  |   |  |

### Citations

1. Datasets

* Chen, Zhuo et al. “Continuous Speech Separation: Dataset and Analysis.” ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2020): 7284-7288.

* McCowan, Iain et al. “The AMI meeting corpus.” (2005).

* Fu, Yihui et al. “AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario.” ArXiv abs/2104.03603 (2021): n. pag.

2. VAD and Overlap detection

* Bredin, Hervé et al. “Pyannote. Audio: Neural Building Blocks for Speaker Diarization.” ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2020): 7124-7128.

* Bredin, Hervé and Antoine Laurent. “End-to-end speaker segmentation for overlap-aware resegmentation.” ArXiv abs/2104.04045 (2021): n. pag.

3. VBx

* Landini, Federico et al. “Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks.” ArXiv abs/2012.14952 (2020)

4. VBx with overlaps

* Bullock, Latané et al. “Overlap-Aware Diarization: Resegmentation Using Neural End-to-End Overlapped Speech Detection.” ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2020): 7114-7118.

5. Spectral clustering

* Park, Tae Jin et al. “Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap.” IEEE Signal Processing Letters 27 (2020): 381-385.

6. Overlap-aware spectral clustering

* Raj, Desh et al. “Multi-Class Spectral Clustering with Overlaps for Speaker Diarization.” 2021 IEEE Spoken Language Technology Workshop (SLT) (2021): 582-589.
