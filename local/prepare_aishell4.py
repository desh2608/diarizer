#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for AISHELL-4 dataset.
import shutil
import random
from pathlib import Path

from lhotse.recipes import prepare_aishell4
from lhotse.manipulation import combine

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
    # Set random seed
    random.seed(0)

    manifests = prepare_aishell4(data_dir)

    # Combine train manifests into one
    train_dev_manifests = {
        "recordings": combine(
            manifests["train_L"]["recordings"],
            manifests["train_M"]["recordings"],
            manifests["train_S"]["recordings"],
        ),
        "supervisions": combine(
            manifests["train_L"]["supervisions"],
            manifests["train_M"]["supervisions"],
            manifests["train_S"]["supervisions"],
        ),
    }
    del manifests["train_L"]
    del manifests["train_M"]
    del manifests["train_S"]

    # Randomly split train manifests into train and dev (20 recordings for dev)
    dev_recording_ids = random.choices(
        list(train_dev_manifests["recordings"].ids), k=20
    )
    dev_recordings = train_dev_manifests["recordings"].filter(
        lambda r: r.id in dev_recording_ids
    )
    dev_supervisions = train_dev_manifests["supervisions"].filter(
        lambda s: s.recording_id in dev_recording_ids
    )
    train_recordings = train_dev_manifests["recordings"].filter(
        lambda r: r.id not in dev_recording_ids
    )
    train_supervisions = train_dev_manifests["supervisions"].filter(
        lambda s: s.recording_id not in dev_recording_ids
    )

    manifests["train"] = {
        "recordings": train_recordings,
        "supervisions": train_supervisions,
    }
    manifests["dev"] = {"recordings": dev_recordings, "supervisions": dev_supervisions}

    output_dir = Path(output_dir)

    for split in ("train", "dev", "test"):
        split_dir = output_dir / split
        audio_dir = split_dir / "audios"
        vad_dir = split_dir / "vad"
        rttm_dir = split_dir / "rttm"
        # Create output directories.
        audio_dir.mkdir(parents=True, exist_ok=True)
        vad_dir.mkdir(parents=True, exist_ok=True)
        rttm_dir.mkdir(parents=True, exist_ok=True)

        # Write audios
        logging.info(f"Preparing {split} audios...")
        for recording in tqdm(manifests[split]["recordings"]):
            recording_id = recording.id
            audio_path = audio_dir / f"{recording_id}.wav"
            x = torch.tensor(recording.load_audio(channels=0))
            torchaudio.save(audio_path, x, 16000)

        # Write RTTM and VAD
        data_dir = Path(data_dir)
        for recording in tqdm(manifests[split]["recordings"]):
            file = next(data_dir.rglob(f"{recording.id}.rttm"))
            shutil.copy(file, rttm_dir)
            # Write VAD
            vad_segments = rttm_to_vad_segments(file)
            with open(vad_dir / f"{recording.id}.lab", "w") as f:
                for segs in vad_segments:
                    start, end = segs
                    f.write(f"{start:.3f} {end:.3f} sp\n")


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.output_dir)
