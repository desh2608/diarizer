#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for AMI dataset. We use the references from BUT's AMI setup:
# https://github.com/BUTSpeechFIT/AMI-diarization-setup
from pathlib import Path
import shutil

from lhotse.recipes import prepare_ami
from tqdm import tqdm
import logging


from utils import supervision_to_vad_segments

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="AMI dataset preparation.")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to AMI data directory."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory."
    )
    return parser.parse_args()


def main(data_dir, output_dir):
    data_dir = Path(data_dir)
    manifests = prepare_ami(
        data_dir,
        annotations_dir=data_dir / "ami_public_manual_1.6.2",
        mic="sdm",
        partition="full-corpus-asr",
    )

    output_dir = Path(output_dir)

    for split in ("train", "dev", "test"):
        logging.info(f"Processing {split} split.")

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        audio_dir = split_dir / "audios"
        vad_dir = split_dir / "vad"
        rttm1_dir = split_dir / "rttm_but"
        rttm2_dir = split_dir / "rttm"
        # Create output directories.
        audio_dir.mkdir(parents=True, exist_ok=True)
        vad_dir.mkdir(parents=True, exist_ok=True)
        rttm1_dir.mkdir(parents=True, exist_ok=True)
        rttm2_dir.mkdir(parents=True, exist_ok=True)

        # Write audios
        logging.info("Preparing audios...")
        for recording in tqdm(manifests[split]["recordings"]):
            recording_id = recording.id
            audio_path = audio_dir / f"{recording_id}.wav"
            # Save symlink to audio file.
            if not audio_path.exists():
                audio_path.symlink_to(recording.sources[0].source)

        # Copy RTTM and VAD from BUT's AMI setup.
        for file in Path(f"AMI-diarization-setup/only_words/labs/{split}").iterdir():
            shutil.copy(file, vad_dir)
        for file in Path(f"AMI-diarization-setup/only_words/rttms/{split}").iterdir():
            shutil.copy(file, rttm1_dir)

        # Write RTTM from supervisions
        rttm_string = "SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        with open(rttm2_dir / f"{recording_id}.rttm", "w") as f:
            for supervision in manifests[split]["supervisions"]:
                start = supervision.start
                duration = supervision.duration
                speaker = supervision.speaker
                f.write(rttm_string.format(**locals()))
                f.write("\n")

        recording_list = set(manifests[split]["recordings"].ids)
        vad_list = set(Path(vad_dir).glob("*.lab"))
        vad_list = {vad.stem for vad in vad_list}
        assert recording_list == vad_list, "VAD and recording lists are not equal."


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.output_dir)
