#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for LibriCSS dataset.
from pathlib import Path
from itertools import groupby

from lhotse.recipes import prepare_libricss
import lhotse
from tqdm import tqdm
import logging

import torch
import torchaudio

from utils import supervision_to_vad_segments

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="LibriCSS dataset preparation.")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to LibriCSS data directory."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory."
    )
    return parser.parse_args()


def main(data_dir, output_dir):
    manifests = prepare_libricss(data_dir)

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
    for recording in tqdm(manifests["recordings"]):
        recording_id = recording.id
        audio_path = audio_dir / f"{recording_id}.wav"
        x = torch.tensor(recording.load_audio(channels=0))
        torchaudio.save(audio_path, x, 16000)

    # Write RTTM and VAD
    rttm_string = "SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
    reco_to_supervision = groupby(
        sorted(manifests["supervisions"], key=lambda seg: seg.recording_id),
        key=lambda seg: seg.recording_id,
    )
    logging.info("Preparing RTTM and VAD...")
    for recording_id, supervisions in tqdm(reco_to_supervision):
        supervisions = list(supervisions)
        supervisions.sort(key=lambda seg: seg.start)
        # Write RTTM
        with open(rttm_dir / f"{recording_id}.rttm", "w") as f:
            for supervision in supervisions:
                start = supervision.start
                duration = supervision.duration
                speaker = supervision.speaker
                f.write(rttm_string.format(**locals()))
                f.write("\n")
        # Write VAD
        vad_segments = supervision_to_vad_segments(supervisions)
        with open(vad_dir / f"{recording_id}.lab", "w") as f:
            for segs in vad_segments:
                start, end = segs
                f.write(f"{start:.3f} {end:.3f} sp\n")


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.output_dir)
