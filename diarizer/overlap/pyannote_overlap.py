#!/usr/local/env/python3
import argparse
from pathlib import Path

from pyannote.audio.pipelines import OverlappedSpeechDetection


def get_args():
    parser = argparse.ArgumentParser(
        description="Run Pyannote speech activity detection."
    )
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
        help="Path to the output directory where the label file will be written.",
    )
    parser.add_argument("--onset", type=float, default=0.5, help="Onset threshold.")
    parser.add_argument("--offset", type=float, default=0.5, help="Offset threshold.")
    parser.add_argument(
        "--min-duration-on",
        type=float,
        default=0.0,
        help="Remove speech regions shorter than that many seconds.",
    )
    parser.add_argument(
        "--min-duration-off",
        type=float,
        default=0.0,
        help="Fill non-speech regions shorter than that many seconds.",
    )
    return parser.parse_args()


def main(in_dir, files, out_dir, HYPER_PARAMETERS):
    out_dir.mkdir(exist_ok=True, parents=True)

    ovl_pipeline = OverlappedSpeechDetection(
        segmentation="pyannote/segmentation", device="cpu"
    )
    ovl_pipeline.instantiate(HYPER_PARAMETERS)

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

    HYPER_PARAMETERS = {
        "onset": args.onset,
        "offset": args.offset,
        "min_duration_on": args.min_duration_on,
        "min_duration_off": args.min_duration_off,
    }

    main(in_dir, files, out_dir, HYPER_PARAMETERS)
