#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for CALLHOME dataset.
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

# fmt:off
TWO_SPEAKER_RECORDINGS = ["iaaa","iaac","iaal","iaan","iaat","iaau","iaax","iabb","iabc",
    "iabf","iabk","iabl","iabm","iabr","iabs","iabx","iacc","iaci","iacm","iaco","iacq",
    "iacr","iacs","iacx","iada","iadg","iadu","iady","iadz","iaeb","iaed","iaee","iaef",
    "iaei","iaek","iael","iaen","iaep","iaes","iafj","iafk","iafo","iafq","iafz","iagb",
    "iagh","iagi","iagn","iahg","iahi","iahj","iaho","iahq","iahr","iahv","iahw","iahy",
    "iaia","iaib","iaid","iaie","iaik","iaiu","iaiz","iajd","iajf","iajh","iajo","iajp",
    "iajt","iajy","iajz","iakh","iakj","iakq","iaks","iakt","iakw","iakx","iaky","iakz",
    "iald","ialg","ialm","ialy","iamb","iamc","iamj","iamm","iamn","iamo","iamq","iamx",
    "iamy","iana","ianc","iane","iani","ianl","ianm","ianw","iany","ianz","iaod","iaof",
    "iaok","iaol","iaoo","iaoq","iaor","iaot","iaow","iaox","iape","iapf","iaph","iapi",
    "iapk","iapm","iapn","iapp","iapr","iapw","iapy","iaqa","iaqe","iaqh","iaqi","iaqj",
    "iaql","iaqm","iaqn","iaqu","iarg","iark","iarr","iars","iaru","iarv","iasc","iasd",
    "iase","iasg","iasi","iask","iasm","iaso","iasp","iasq","iasr","iast","iasu","iasz",
    "iatc","iate","iaaf","iaai","iaam","iaaq","iaar","iaaw","iaay","iabe","iabj","iabo",
    "iabp","iabt","iabv","iabw","iaby","iaca","iacd","iach","iacj","iacl","iacp","iact",
    "iacv","iacw","iacy","iacz","iadb","iadc","iadd","iade","iadf","iadl","iadn","iads",
    "iadv","iaea","iaeg","iaeh","iaej","iaeo","iaev","iaew","iaey","iafa","iafc","iafd",
    "iafg","iafi","iafl","iafm","iaft","iafv","iafw","iafy","iagd","iagf","iagk","iagl",
    "iago","iagq","iaha","iahc","iahh","iahl","iahm","iahn","iaht","iahx","iahz","iaic",
    "iaih","iaim","iain","iair","iaix","iaja","iajc","iaje","iaji","iajl","iajm","iajv",
    "iajw","iajx","iakb","iakc","iakd","iakl","iakm","iala","ialc","iale","iali","ialo",
    "ialp","ialq","ialr","ials","ialt","ialw","ialx","iami","iamk","iamp","iamr","iamv",
    "iamz","ianf","ianh","iaob","iaoe","iaog","iaoi","iaoj","iaom","iaou","iapa","iapb",
    "iapc","iapg","iaqd","iaqo","iaqp","iaqq","iaqr","iaqt","iaqv","iaqy","iara","iarb",
    "iarc","iare","iarp","iart","iarw","iarx","iary","iarz","iasb","iasf","iasl","iasn",
    "iass","iasv","iasx","iasy","iata","iatf"]
# fmt:on


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="CALLHOME dataset preparation.")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to CALLHOME data directory."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory."
    )
    parser.add_argument(
        "--two-speakers",
        action="store_true",
        help="Prepare only 2-speaker subset of the data.",
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

    if args.two_speakers:
        manifests["recordings"] = manifests["recordings"].filter(
            lambda r: r.id in TWO_SPEAKER_RECORDINGS
        )
        manifests["supervisions"] = manifests["supervisions"].filter(
            lambda s: s.recording_id in TWO_SPEAKER_RECORDINGS
        )

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

            if args.two_speakers and recording_id not in TWO_SPEAKER_RECORDINGS:
                continue

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
