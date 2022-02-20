#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for LibriCSS dataset.
from os import sep
from pathlib import Path
from itertools import groupby

from lhotse.recipes import prepare_libricss

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

SESSIONS = {
    "dev": ["session0"],
    "test": [
        "session1",
        "session2",
        "session3",
        "session4",
        "session5",
        "session6",
        "session7",
        "session8",
        "session9",
    ],
}


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


def main(data_dir, out_dir):
    manifests = prepare_libricss(data_dir)

    for part in ["dev", "test"]:
        output_dir = Path(out_dir) / part
        audio_dir = output_dir / "audios"
        vad_dir = output_dir / "vad"
        rttm_dir = output_dir / "rttm"
        # Create output directories.
        audio_dir.mkdir(parents=True, exist_ok=True)
        vad_dir.mkdir(parents=True, exist_ok=True)
        rttm_dir.mkdir(parents=True, exist_ok=True)

        # Write audios
        logging.info(f"Preparing {part} audios...")
        for recording in tqdm(
            filter(
                lambda r: any(session in r.id for session in SESSIONS[part]),
                manifests["recordings"],
            )
        ):
            recording_id = recording.id
            if all([session not in recording_id for session in SESSIONS[part]]):
                continue
            audio_path = audio_dir / f"{recording_id}.wav"
            x = torch.tensor(recording.load_audio(channels=0))
            torchaudio.save(audio_path, x, 16000)

        # Write RTTM and VAD
        rttm_string = "SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        reco_to_supervision = groupby(
            sorted(manifests["supervisions"], key=lambda seg: seg.recording_id),
            key=lambda seg: seg.recording_id,
        )
        logging.info(f"Preparing {part} RTTM and VAD...")
        for recording_id, supervisions in tqdm(reco_to_supervision):
            if all([session not in recording_id for session in SESSIONS[part]]):
                continue
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
