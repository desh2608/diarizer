#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for LibriCSS dataset.
from collections import defaultdict
from pathlib import Path
from itertools import groupby, chain
import sys

from lhotse.recipes import prepare_libricss
from lhotse import RecordingSet, Recording, CutSet

from tqdm import tqdm
import logging

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


def file_name_to_session_and_channel(file_name, oracle=False):
    """
    Extract session and channel from file name.
    """
    if not oracle:
        _, _, ovl, _, sil, session, _, _, channel = file_name.split("_")
        ovl = int(float(ovl))
        if ovl == 0:
            if sil == "0.5":
                session = f"0S_{session}"
            else:
                session = f"0L_{session}"
        else:
            session = f"OV{ovl}_{session}"
        return session, int(channel)
    else:
        session, channel = file_name.rsplit("_", 1)
        return session, int(channel)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="LibriCSS dataset preparation.")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to LibriCSS data directory."
    )
    parser.add_argument(
        "--separated-dir",
        type=str,
        required=True,
        help="Path to directory containing separated wav files.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory."
    )
    parser.add_argument("--oracle", dest="oracle_css", action="store_true")
    return parser.parse_args()


def main(args, data_dir, out_dir, separated_dir):
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
        manifests = prepare_libricss(data_dir, type="ihm")
        cuts = CutSet.from_manifests(
            recordings=manifests["recordings"], supervisions=manifests["supervisions"]
        ).trim_to_supervisions(keep_overlapping=False)
        get_oracle_css(cuts, separated_dir)

    manifests = prepare_libricss(data_dir, type="mdm")
    separated_recordings = []
    reco2channels = defaultdict(list)
    for audio in separated_dir.rglob("*.wav"):
        session, channel = file_name_to_session_and_channel(
            audio.stem, oracle=args.oracle_css
        )
        reco2channels[session].append(f"{session}_{channel}")
        separated_recordings.append(
            Recording.from_file(audio, recording_id=f"{session}_{channel}")
        )
    manifests["separated"] = RecordingSet.from_recordings(separated_recordings)

    for part in ["dev", "test"]:
        output_dir = Path(out_dir) / part
        audio_dir = output_dir / "audios"
        vad_dir = output_dir / "vad"
        rttm_dir = output_dir / "rttm"
        reco2channel_file = output_dir / "reco2channel"

        # Create output directories.
        audio_dir.mkdir(parents=True, exist_ok=True)
        vad_dir.mkdir(parents=True, exist_ok=True)
        rttm_dir.mkdir(parents=True, exist_ok=True)

        # Write reco2channel file
        with open(reco2channel_file, "w") as f:
            for reco in sorted(reco2channels):
                if any(session in reco for session in SESSIONS[part]):
                    f.write(f"{reco} {' '.join(reco2channels[reco])}\n")

        # Write audios
        for recording in tqdm(
            filter(
                lambda r: any(session in r.id for session in SESSIONS[part]),
                manifests["separated"],
            ),
            desc=f"Writing separated audios for {part}",
        ):
            recording_id = recording.id
            audio_path = audio_dir / f"{recording_id}.wav"
            audio_path.symlink_to(Path(recording.sources[0].source).resolve())

        # Write RTTM and VAD
        rttm_string = "SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        reco_to_supervision = groupby(
            sorted(manifests["supervisions"], key=lambda seg: seg.recording_id),
            key=lambda seg: seg.recording_id,
        )
        for recording_id, supervisions in tqdm(
            reco_to_supervision, desc=f"Writing RTTM and VAD for {part}"
        ):
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
    main(args, args.data_dir, args.output_dir, args.separated_dir)
