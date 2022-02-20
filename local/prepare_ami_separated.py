#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for AMI dataset. We use the references from BUT's AMI setup:
# https://github.com/BUTSpeechFIT/AMI-diarization-setup
from pathlib import Path
from collections import defaultdict
import shutil
import sys

from lhotse.recipes import prepare_ami
from lhotse import CutSet, Recording, RecordingSet
from lhotse.manipulation import combine
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
    parser.add_argument(
        "--separated-dir",
        type=str,
        required=True,
        help="Path to directory containing separated wav files.",
    )
    parser.add_argument("--oracle", dest="oracle_css", action="store_true")
    return parser.parse_args()


def main(args, data_dir, output_dir, separated_dir):
    data_dir = Path(data_dir)
    separated_dir = Path(separated_dir)

    if not separated_dir.exists():
        from diarizer.data_utils import get_oracle_css

        if not args.oracle_css:
            logging.info(
                "Separated directory not found. If you want to use oracle css, please set --oracle flag."
            )
            sys.exit(1)

        logging.info(
            "Separated directory does not exist. Creating oracle from headset mics."
        )
        separated_dir.mkdir(parents=True, exist_ok=True)
        manifests = prepare_ami(
            data_dir,
            annotations_dir=data_dir / "ami_public_manual_1.6.2",
            mic="ihm",
            partition="full-corpus-asr",
        )
        cuts = combine(
            CutSet.from_manifests(
                recordings=manifests[split]["recordings"],
                supervisions=manifests[split]["supervisions"],
            ).trim_to_supervisions(keep_overlapping=False)
            for split in ["dev", "test"]
        )
        get_oracle_css(cuts, separated_dir)

    manifests = prepare_ami(
        data_dir,
        annotations_dir=data_dir / "ami_public_manual_1.6.2",
        mic="sdm",
        partition="full-corpus-asr",
    )

    separated_recordings = []
    reco2channels = defaultdict(list)
    for audio in separated_dir.rglob("*.wav"):
        session, channel = audio.stem.split("_")
        reco2channels[session].append(f"{session}_{channel}")
        separated_recordings.append(
            Recording.from_file(audio, recording_id=f"{session}_{channel}")
        )
    manifests["separated"] = RecordingSet.from_recordings(separated_recordings)

    output_dir = Path(output_dir)

    for split in ("dev", "test"):
        logging.info(f"Processing {split} split.")

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        audio_dir = split_dir / "audios"
        vad_dir = split_dir / "vad"
        rttm1_dir = split_dir / "rttm_but"
        rttm2_dir = split_dir / "rttm"
        reco2channel_file = split_dir / "reco2channel"

        # Create output directories.
        audio_dir.mkdir(parents=True, exist_ok=True)
        vad_dir.mkdir(parents=True, exist_ok=True)
        rttm1_dir.mkdir(parents=True, exist_ok=True)
        rttm2_dir.mkdir(parents=True, exist_ok=True)

        # Write reco2channel file
        with open(reco2channel_file, "w") as f:
            for reco in manifests[split]["recordings"].ids:
                f.write(f"{reco} {' '.join(reco2channels[reco])}\n")

        # Write audios
        for recording in tqdm(
            filter(
                lambda r: any(
                    session in r.id for session in manifests[split]["recordings"].ids
                ),
                manifests["separated"],
            ),
            desc=f"Writing separated audios for {split}",
        ):
            recording_id = recording.id
            audio_path = audio_dir / f"{recording_id}.wav"
            audio_path.symlink_to(Path(recording.sources[0].source).resolve())

        # Copy RTTM and VAD from BUT's AMI setup.
        for file in Path(f"AMI-diarization-setup/only_words/labs/{split}").iterdir():
            shutil.copy(file, vad_dir)
        for file in Path(f"AMI-diarization-setup/only_words/rttms/{split}").iterdir():
            shutil.copy(file, rttm1_dir)

        # Write RTTM from supervisions
        rttm_string = "SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        for recording_id in tqdm(manifests[split]["recordings"].ids):
            with open(rttm2_dir / f"{recording_id}.rttm", "w") as f:
                for supervision in manifests[split]["supervisions"].filter(
                    lambda s: s.recording_id == recording_id
                ):
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
    main(args, args.data_dir, args.output_dir, args.separated_dir)
