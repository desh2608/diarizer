## Clustering-based Diarization

Python implementations of some clustering-based diarization systems.

### Features

* End-to-end recipes (from unsegmented audio to evaluation) for LibriCSS, AMI, AISHELL-4, and AliMeeting.
* Using [Lhotse](https://github.com/lhotse-speech/lhotse) for data preparation. 
* Using [Pyannote 2.0](https://github.com/pyannote/pyannote-audio/tree/develop) models for VAD and overlap detection.
* Scripts for fine-tuning Pyannote models on AMI, AISHELL-4, and AliMeeting (fine-tuned models also provided).
* VBx and x-vector extraction from [BUT](https://github.com/BUTSpeechFIT/VBx)'s implementation.
* [Kaldi](https://github.com/kaldi-asr/kaldi) implementation of overlap-aware spectral clustering.

### Installation

We recommend installation within a virtual environment (such as Conda) so that the
package versions are consistent. To create a new Conda environment:

```
> conda create -n diar python=3.8
> conda activate diar
```

Once the environment is activated, clone and install the package as:

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

End-to-end runnable recipes are provided in the `scripts` directory. The scripts must be
invoked from the root directory, for example: `scripts/ami/010_prepare_data.sh` .

Each recipe (LibriCSS, AMI, AISHELL-4) contains scripts broken down into stages such as:
data preparation, VAD, x-vector extraction, overlap detection, and clustering, numbered
in order as 010, 020, etc. These scripts are supposed to be run in order.

By default, the scripts submit commands through the `queue.pl` script (see `utils` folder). To change
this behaviour, please modify the `cmd.sh` file according to your job submission system.

### Results

* **Voice activity detection (VAD) using Pyannote**

| Method   | MS    | FA | Total   |
|----------|-------|----|------|
| LibriCSS | 0.9 | 1.2 | 2.1 | 
| AMI | 3.5 | 2.8 | 6.3 |
| AISHELL-4 | 3.3 | 2.3 | 5.6 |
| AliMeeting | 1.9 | 1.8 | 3.7 |

* **Speaker diarization (using above VAD)**

The following is evaluated using the [spyder](https://github.com/desh2608/spyder) package without ignoring overlaps and using a 0.0 collar.

1. **LibriCSS**

| Method   | MS    | FA | Conf. | DER   | RTTM |
|----------|-------|----|-------|-------|------|
| VBx | 10.37 | 1.19 | 2.96 | 14.52 | [link](rttm/libricss_test_vbx.tar.gz) |
| VBx + OVL | 3.39 | 2.31 | 5.55 | 11.25 | [link](rttm/libricss_test_vbx_ovl.tar.gz) |
| Spectral | 10.37 | 1.19 | 3.37 | 14.93 |[link](rttm/libricss_test_spectral.tar.gz) |
| Spectral + OVL | 3.79 | 2.22 | 5.33 | 11.34 |[link](rttm/libricss_test_spectral_ovl.tar.gz) |

2. **AMI (SDM)**

| Method   | MS    | FA | Conf. | DER   | RTTM |
|----------|-------|----|-------|-------|------|
| VBx | 18.15 | 3.24 | 4.83 | 26.22 | [link](rttm/ami_test_vbx.tar.gz) |
| VBx + OVL | 9.04 | 7.70 | 8.31  | 25.05 | [link](rttm/ami_test_vbx_ovl.tar.gz) |
| Spectral | 18.15 | 3.24 | 4.14 | 25.53 | [link](rttm/ami_test_spectral.tar.gz) |
| Spectral + OVL | 9.63 | 7.39 | 6.67 | 23.69 | [link](rttm/ami_test_spectral_ovl.tar.gz) |

3. **AISHELL-4**

| Method   | MS    | FA | Conf. | DER   | RTTM |
|----------|-------|----|-------|-------|------|
| VBx | 8.27 | 2.80 | 6.94 | 18.01 | [link](rttm/aishell4_test_vbx.tar.gz) |
| VBx + OVL | 5.78 | 7.88 | 7.96 | 21.62 | [link](rttm/aishell4_test_vbx_ovl.tar.gz) |
| Spectral | 8.27 | 2.80 | 5.06 | 16.13 | [link](rttm/aishell4_test_spectral.tar.gz) |
| Spectral + OVL | 5.90 | 7.85 | 5.94 | 19.69 | [link](rttm/aishell4_test_spectral_ovl.tar.gz) |

4. **AliMeeting** (collar=0.25)

| Method   | MS    | FA | Conf. | DER   | RTTM |
|----------|-------|----|-------|-------|------|
| VBx | 13.60 | 0.17 | 3.01 | 16.78 | [link](rttm/alimeeting_test_vbx.tar.gz) |
| VBx + OVL | 5.87 | 2.82 | 5.36 | 14.05 | [link](rttm/alimeeting_test_vbx_ovl.tar.gz) |
| Spectral | 13.60 | 0.17 | 2.60 | 16.37 | [link](rttm/alimeeting_test_spectral.tar.gz) |
| Spectral + OVL | 5.96 | 2.83 | 5.64 | 14.43 | [link](rttm/alimeeting_test_spectral_ovl.tar.gz) |

### Citations

1. **Datasets**

* Chen, Zhuo et al. “Continuous Speech Separation: Dataset and Analysis.” ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2020): 7284-7288.

* McCowan, Iain et al. “The AMI meeting corpus.” (2005).

* Fu, Yihui et al. “AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario.” ArXiv abs/2104.03603 (2021): n. pag.

* Yu, Fan et al. “M2MeT: The ICASSP 2022 Multi-Channel Multi-Party Meeting Transcription Challenge.” ArXiv abs/2110.07393 (2021).

2. **VAD and Overlap detection**

* Bredin, Hervé et al. “Pyannote. Audio: Neural Building Blocks for Speaker Diarization.” ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2020): 7124-7128.

* Bredin, Hervé and Antoine Laurent. “End-to-end speaker segmentation for overlap-aware resegmentation.” ArXiv abs/2104.04045 (2021): n. pag.

3. **VBx**

* Landini, Federico et al. “Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks.” ArXiv abs/2012.14952 (2020)

4. **VBx with overlaps**

* Bullock, Latané et al. “Overlap-Aware Diarization: Resegmentation Using Neural End-to-End Overlapped Speech Detection.” ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2020): 7114-7118.

5. **Spectral clustering**

* Park, Tae Jin et al. “Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap.” IEEE Signal Processing Letters 27 (2020): 381-385.

6. **Overlap-aware spectral clustering**

* Raj, Desh et al. “Multi-Class Spectral Clustering with Overlaps for Speaker Diarization.” 2021 IEEE Spoken Language Technology Workshop (SLT) (2021): 582-589.
