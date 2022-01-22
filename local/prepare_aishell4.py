#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for AISHELL-4 dataset.
import shutil
from pathlib import Path
from itertools import groupby

from lhotse.recipes import prepare_aishell4

from tqdm import tqdm
import logging

from utils import rttm_to_vad_segments

import torch
import torchaudio

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="AISHELL-4 dataset preparation.")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to AISHELL-4 data directory."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory."
    )
    return parser.parse_args()


def main(data_dir, output_dir):
    manifests = prepare_aishell4(data_dir)
    # Keep only test split
    # manifests = manifests["test"]

    output_dir = Path(output_dir)
    audio_dir = output_dir / "audios"
    vad_dir = output_dir / "vad"
    rttm_dir = output_dir / "rttm"
    # Create output directories.
    audio_dir.mkdir(parents=True, exist_ok=True)
    vad_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    # Write audios
    logging.info("Preparing audios...")
    for split in ["train_L", "train_M", "train_S"]:
        for recording in tqdm(manifests[split]["recordings"]):
            recording_id = recording.id
            audio_path = audio_dir / f"{recording_id}.wav"
            x = torch.tensor(recording.load_audio(channels=0))
            torchaudio.save(audio_path, x, 16000)

    # Write RTTM and VAD
    data_dir = Path(data_dir)
    for file in (data_dir / "test" / "TextGrid").rglob("*.rttm"):
        shutil.copy(file, rttm_dir)
        recording_id = file.stem
        # Write VAD
        vad_segments = rttm_to_vad_segments(file)
        with open(vad_dir / f"{recording_id}.lab", "w") as f:
            for segs in vad_segments:
                start, end = segs
                f.write(f"{start:.3f} {end:.3f} sp\n")


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.output_dir)
