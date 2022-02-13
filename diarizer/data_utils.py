# This provides some utility functions for generating/processing data.
from itertools import groupby
from tqdm import tqdm
import random

import numpy as np
import soundfile as sf

from lhotse.utils import compute_num_samples


def get_oracle_css(cuts, out_dir):
    """
    Given a Lhotse cut set containing headset microphone segments, combine them for
    each recording into non-overlapping channels and save the new 2-channel recordings
    to the output directory with the suffix _0 and _1.
    """
    # Group cuts by recording.
    cuts_by_recording = groupby(
        sorted(cuts, key=lambda c: c.recording_id), key=lambda c: c.recording_id
    )
    # Process each recording.
    for recording_id, cuts in tqdm(cuts_by_recording):
        cuts = sorted(cuts, key=lambda c: c.start)
        num_samples = compute_num_samples(
            cuts[0].recording.duration, cuts[0].sampling_rate
        )
        ch0_wav = np.zeros((1, num_samples))
        ch1_wav = np.zeros((1, num_samples))
        prev_ch0, prev_ch1 = 0, 0
        for cut in cuts:
            st = compute_num_samples(cut.start, cut.sampling_rate)
            et = compute_num_samples(cut.end, cut.sampling_rate)
            if (cut.start >= prev_ch0 and cut.start >= prev_ch1) or (
                cut.start < prev_ch0 and cut.start < prev_ch1
            ):
                # if new utterance starts before or after both utterances on last channel
                # assign it randomly to one of the two channels
                if random.random() > 0.5:
                    ch0_wav[:, st:et] += cut.load_audio()
                    prev_ch0 = max(prev_ch0, cut.end)
                else:
                    ch1_wav[:, st:et] += cut.load_audio()
                    prev_ch1 = max(prev_ch1, cut.end)
            elif cut.start >= prev_ch0:
                ch0_wav[:, st:et] += cut.load_audio()
                prev_ch0 = max(prev_ch0, cut.end)
            elif cut.start >= prev_ch1:
                ch1_wav[:, st:et] += cut.load_audio()
                prev_ch1 = max(prev_ch1, cut.end)
        # Save the two-channel audio.
        sf.write(
            file=str(out_dir / f"{recording_id}_0.wav"),
            data=ch0_wav.transpose(),
            samplerate=cut.sampling_rate,
        )
        sf.write(
            file=str(out_dir / f"{recording_id}_1.wav"),
            data=ch1_wav.transpose(),
            samplerate=cut.sampling_rate,
        )
