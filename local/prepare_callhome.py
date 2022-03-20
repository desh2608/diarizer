#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for AISHELL-4 dataset.
import random
import urllib
from pathlib import Path

from lhotse import load_manifest
from lhotse.recipes.callhome_english import prepare_callhome_english_sre

from tqdm import tqdm
import soundfile as sf
import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

SPLIT_BASE_URL = "https://raw.githubusercontent.com/BUTSpeechFIT/VBx/master/data/CALLHOME/lists/{part}.txt"
RTTM_BASE_URL = "https://raw.githubusercontent.com/BUTSpeechFIT/VBx/master/data/CALLHOME/rttms/all/{recording_id}.rttm"


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="CALLHOME dataset preparation.")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to CALLHOME data directory."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory."
    )
    return parser.parse_args()


def main(data_dir, output_dir):
    # Set random seed
    random.seed(0)

    # Maybe the manifests already exist: we can read them and save a bit of preparation time.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        manifests = {
            "recordings": load_manifest(Path(output_dir) / "recordings.jsonl"),
            "supervisions": load_manifest(Path(output_dir) / "supervisions.jsonl"),
        }
    except:
        manifests = prepare_callhome_english_sre(data_dir, absolute_paths=True)
        manifests["recordings"].to_file(Path(output_dir) / "recordings.jsonl")
        manifests["supervisions"].to_file(Path(output_dir) / "supervisions.jsonl")

    output_dir = Path(output_dir)

    for split in ("dev", "test"):
        split_dir = output_dir / split
        audio8k_dir = split_dir / "audios_8k"
        audio16k_dir = split_dir / "audios_16k"
        vad_dir = split_dir / "vad"
        rttm_dir = split_dir / "rttm"

        # Create output directories.
        audio8k_dir.mkdir(parents=True, exist_ok=True)
        audio16k_dir.mkdir(parents=True, exist_ok=True)
        vad_dir.mkdir(parents=True, exist_ok=True)
        rttm_dir.mkdir(parents=True, exist_ok=True)

        # We will use part1 as dev and part2 as test.
        part = "part1" if split == "dev" else "part2"
        target_url = SPLIT_BASE_URL.format(part=part)

        rttm_string = "SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        for line in tqdm(urllib.request.urlopen(target_url), desc=split):
            recording_id = line.decode("utf-8").strip()

            # Write RTTM from supervisions
            with open(rttm_dir / f"{recording_id}.rttm", "w") as f:
                for supervision in manifests["supervisions"].filter(
                    lambda s: s.recording_id == recording_id
                ):
                    start = supervision.start
                    duration = supervision.duration
                    speaker = supervision.speaker
                    f.write(rttm_string.format(**locals()))
                    f.write("\n")

            audio8k_path = audio8k_dir / f"{recording_id}.wav"
            audio16k_path = audio16k_dir / f"{recording_id}.wav"
            recording_8k = manifests["recordings"][recording_id]
            recording_16k = recording_8k.resample(16000)

            try:
                x = recording_8k.load_audio()
                sf.write(audio8k_path, x.T, 8000, format="FLAC")
            except:
                logging.warning(f"Failed to load 8k {recording_id}.")

            try:
                x = recording_16k.load_audio()
                sf.write(audio16k_path, x.T, 16000, format="FLAC")
            except Exception as e:
                logging.warning(f"Failed to load 16k {recording_id}.\n{e}")


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.output_dir)
