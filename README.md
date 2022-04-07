## InterSpeech 2022: Baselines for telephone conversation datasets

This branch contains scripts for reproducing the clustering-based baseline systems
for the paper: **Leveraging Speech Separation for Conversational Telephone Speaker Diarization**, 
submitted to InterSpeech 2022. Scripts are provided for the real Fisher and CALLHOME
datasets.

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

To run the VAD stages, we will additionally need Kaldi. Clone and install Kaldi, and
then set the `KALDI_ROOT` in path.sh file to your Kaldi location. You would also need
to create symbolic links to `steps` and `utils` inside the corresponding scripts directory.

```
> cd scripts/callhome
> KALDI_ROOT=/path/to/kaldi
> ln -s $KALDI_ROOT/egs/wsj/s5/utils .
> ln -s $KALDI_ROOT/egs/wsj/s5/steps .
```

### Usage

End-to-end runnable recipes are provided in the `scripts` directory. The scripts must be
invoked from the root directory, for example: `scripts/callhome/010_prepare_data.sh` .

Each recipe (fisher, callhome) contains scripts broken down into stages such as:
data preparation, VAD, x-vector extraction, overlap detection, and clustering, numbered
in order as 010, 020, etc. These scripts are supposed to be run in order.

### Results

The following is evaluated using `md-eval.pl` without ignoring overlaps and using a 0.25 collar.

1. **Fisher**

| Method   | MS    | FA | Conf. | DER   |
|----------|-------|----|-------|-------|
| VBx | 8.88 | 0.42 | 0.93 | 10.23 |
| VBx + OVL | 4.35 | 2.12 | 0.88 | 7.35 |
| Spectral | 8.88 | 0.42 | 0.22 | 9.52 |
| Spectral + OVL | 5.19 | 1.99 | 0.18 | 7.36 |

2. **CALLHOME**

| Method   | MS    | FA | Conf. | DER   |
|----------|-------|----|-------|-------|
| VBx | 8.28 | 0.87 | 2.59 | 11.74 |
| VBx + OVL | 5.34 | 2.48 | 2.43  | 10.25 |
| Spectral | 8.28 | 0.87 | 5.31 | 14.46 |
| Spectral + OVL | 5.70 | 2.67 | 5.76 | 14.13 |

### Citations

```
@inproceedings{Morrone2022LeveragingSS,
  title={Leveraging Speech Separation for Conversational Telephone Speaker Diarization},
  author={Giovanni Morrone and Samuele Cornell and Desh Raj and Enrico Zovato and Alessio Brutti and Stefano Squartini},
  year={2022}
}
```
