#!/usr/local/env/python3
import argparse
import requests
import zipfile
from pathlib import Path

from pyannote.audio.features import Pretrained as _Pretrained
from pyannote.audio.pipeline.overlap_detection import OverlapDetection


PYANNOTE_AUDIO_HUB_BASE_URL = (
    "https://github.com/pyannote/pyannote-audio-hub/blob/master/models/"
)


def get_args():
    parser = argparse.ArgumentParser(description="Run Pyannote overlap detection.")
    parser.add_argument(
        "--in-dir",
        type=str,
        help="Path to the input directory containing the wav files.",
    )
    parser.add_argument(
        "--file-list",
        type=str,
        help="List of wav files to process.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Path to the output directory where the overlap RTTM will be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model to use for overlap detection.",
        default="ovl_dihard",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        help="Path to save Pyannote models.",
        default="exp/pyannote",
    )
    return parser.parse_args()


def main(in_dir, files, out_dir, exp_dir, model):
    out_dir.mkdir(exist_ok=True, parents=True)
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Download model from pyannote audio hub
    model_url = f"{PYANNOTE_AUDIO_HUB_BASE_URL}/{model}.zip?raw=true"
    pretrained_dir = exp_dir / model
    if not pretrained_dir.exists():
        r = requests.get(model_url)
        with open(exp_dir / f"{model}.zip", "wb") as f:
            f.write(r.content)

        # Unzip model
        with zipfile.ZipFile(exp_dir / f"{model}.zip") as z:
            z.extractall(exp_dir / model)

    # Load model
    (params_yml,) = pretrained_dir.rglob("params.yml")
    pretrained = _Pretrained(
        validate_dir=params_yml.parent,
        duration=None,
        step=0.25,
        batch_size=128,
        device="cpu",
    )
    ovl_pipeline = OverlapDetection(scores=pretrained).load_params(params_yml)

    for file in in_dir.rglob("*.wav"):
        file_id = file.stem
        if file_id not in files:
            continue
        ovl_out = ovl_pipeline({"audio": file})
        with open(f"{out_dir}/{file_id}.rttm", "w") as f:
            ovl_out.write_rttm(f)


if __name__ == "__main__":
    args = get_args()
    in_dir = Path(args.in_dir)
    with open(args.file_list) as f:
        files = [line.strip() for line in f]
    out_dir = Path(args.out_dir)
    exp_dir = Path(args.exp_dir)

    main(in_dir, files, out_dir, exp_dir, args.model)
