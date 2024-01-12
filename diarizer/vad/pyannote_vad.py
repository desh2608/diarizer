#!/usr/local/env/python3
import argparse
from pathlib import Path

from pyannote.audio.pipelines import VoiceActivityDetection


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
    parser.add_argument(
        "--model",
        type=str,
        default="pyannote/segmentation",
        help="Path to the model. If not provided, we use the pretrained model from HuggingFace.",
    )
    parser.add_argument(
        "--use-auth-token",
        type=str,
        default=None,
        help="HuggingFace auth token to use the model from HuggingFace.",
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
    parser.add_argument(
        "--align-time",
        default=None,
        type=float,
        help="If provided, make start and end times multiples of this value.",
    )
    return parser.parse_args()


def main(args, in_dir, files, out_dir, HYPER_PARAMETERS):
    out_dir.mkdir(exist_ok=True, parents=True)

    vad_pipeline = VoiceActivityDetection(
        segmentation=args.model, device="cpu", use_auth_token=args.use_auth_token
    )
    vad_pipeline.instantiate(HYPER_PARAMETERS)

    for file in in_dir.rglob("*.wav"):
        file_id = file.stem
        if file_id not in files:
            continue
        vad_out = vad_pipeline({"audio": file})
        with open(f"{out_dir}/{file_id}.lab", "w") as f:
            for start, end in vad_out.get_timeline():
                if args.align_time is not None:
                    start = round(start / args.align_time) * args.align_time
                    end = round(end / args.align_time) * args.align_time
                f.write(f"{start:.3f} {end:.3f} sp\n")


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

    main(args, in_dir, files, out_dir, HYPER_PARAMETERS)
