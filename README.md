## Clustering-based Diarization

This repository contains Python implementations of some clustering-based diarization
systems, mainly based on spectral clustering and BUT's VBx. We provide scripts
for running the methods on 3 datasets: LibriCSS, AMI, and AISHELL-4.

This code was originally fork of the VBx repository, but has since evolved to include
additional methods such as overlap-aware spectral clustering.

### Installation

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

### Results

1. Voice activity detection (VAD) using Pyannote

| Method   | MS    | FA | Total   |
|----------|-------|----|------|
| LibriCSS |  |  |  | 
| AMI |  |     |  |
| AISHELL-4 |  |    |  |

2. Speaker diarization (using above VAD)

* LibriCSS

| Method   | MS    | FA | Conf. | DER   |
|----------|-------|----|-------|-------|
| VBx |  |  |  | |
| VBx + OVL |  |  |   |  |
| Spectral |  |  |   |  |
| Spectral + OVL |  |  |   |  |

* AMI

| Method   | MS    | FA | Conf. | DER   |
|----------|-------|----|-------|-------|
| VBx |  |  |  | |
| VBx + OVL |  |  |   |  |
| Spectral |  |  |   |  |
| Spectral + OVL |  |  |   |  |

* AISHELL-4

| Method   | MS    | FA | Conf. | DER   |
|----------|-------|----|-------|-------|
| VBx |  |  |  | |
| VBx + OVL |  |  |   |  |
| Spectral |  |  |   |  |
| Spectral + OVL |  |  |   |  |

### Citations

* VAD and Overlap detection

```
Bredin, Hervé et al. “Pyannote.Audio: Neural Building Blocks for Speaker Diarization.” ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2020): 7124-7128.

Bredin, Hervé and Antoine Laurent. “End-to-end speaker segmentation for overlap-aware resegmentation.” ArXiv abs/2104.04045 (2021): n. pag.
```

* VBx

```
Landini, Federico et al. “Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks.” ArXiv abs/2012.14952 (2020)
```

* VBx with overlaps

```
Bullock, Latané et al. “Overlap-Aware Diarization: Resegmentation Using Neural End-to-End Overlapped Speech Detection.” ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2020): 7114-7118.
```

* Spectral clustering

```
Park, Tae Jin et al. “Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap.” IEEE Signal Processing Letters 27 (2020): 381-385.
```

* Overlap-aware spectral clustering

```
Raj, Desh et al. “Multi-Class Spectral Clustering with Overlaps for Speaker Diarization.” 2021 IEEE Spoken Language Technology Workshop (SLT) (2021): 582-589.
```
