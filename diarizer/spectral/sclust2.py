#!/usr/bin/env python

# @Authors: Desh Raj
# @Emails: r.desh26@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This script performs spectral clustering on CSS separated audio streams.

import argparse
import os
import itertools
from collections import namedtuple

import h5py
import kaldi_io
import numpy as np
from scipy.linalg import eigh

from diarizer.diarization_lib import (
    read_xvector_timing_dict,
    l2_norm,
    cos_similarity,
    kaldi_ivector_plda_scoring_dense,
    mkdir_p,
)
from diarizer.kaldi_utils import read_plda
from diarizer.spectral.Spectral_clustering import NME_SpectralClustering


Segment = namedtuple("Segment", ["channel", "start", "end", "label"])


def write_output(fp, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(
            f"SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} "
            f"<NA> <NA> {int(label + 1)} <NA> <NA>{os.linesep}"
        )


def compute_overlap_vector(overlap_rttm, segments):
    if overlap_rttm is not None:
        overlap_segs = []
        with open(overlap_rttm, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                overlap_segs.append(
                    (float(parts[3]), float(parts[3]) + float(parts[4]))
                )
        ol_vec = np.zeros(len(segments))
        overlap_segs.sort(key=lambda x: x[0])
        for i, segment in enumerate(segments):
            start_time, end_time = segment
            is_overlap = get_overlap_decision(overlap_segs, (start_time, end_time))
            if is_overlap:
                ol_vec[i] = 1
    else:
        ol_vec = -1 * np.ones(len(segments))
    return ol_vec


def get_overlap_decision(overlap_segs, subsegment, frac=0.5):
    """Returns true if at least 'frac' fraction of the subsegment lies
    in the overlap_segs."""
    start_time = subsegment[0]
    end_time = subsegment[1]
    dur = end_time - start_time
    total_ovl = 0

    for seg in overlap_segs:
        cur_start, cur_end = seg
        if cur_start >= end_time:
            break
        ovl_start = max(start_time, cur_start)
        ovl_end = min(end_time, cur_end)
        ovl_time = max(0, ovl_end - ovl_start)

        total_ovl += ovl_time

    return total_ovl >= frac * dur


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-rttm-dir",
        required=True,
        type=str,
        help="Directory to store output rttm files",
    )
    parser.add_argument(
        "--xvec-ark-file",
        required=True,
        type=str,
        help="Kaldi ark file with x-vectors from one or more input recordings. "
        "Attention: all x-vectors from one recording must be in one ark file",
    )
    parser.add_argument(
        "--segments-file",
        required=True,
        type=str,
        help="File with x-vector timing info (see diarization_lib.read_xvector_timing_dict)",
    )
    parser.add_argument(
        "--xvec-transform",
        required=True,
        type=str,
        help="path to x-vector transformation h5 file",
    )
    parser.add_argument(
        "--plda-file",
        required=False,
        type=str,
        help="File with PLDA model in Kaldi format (will use cosine similarity if not provided)",
    )
    parser.add_argument(
        "--min-neighbors",
        required=False,
        type=int,
        default=3,
        help="Minimum number of neighbors to threshold similarity matrix",
    )
    parser.add_argument(
        "--max-neighbors",
        required=False,
        type=int,
        default=20,
        help="Maximum number of neighbors to threshold similarity matrix",
    )
    parser.add_argument(
        "--num-speakers",
        required=False,
        type=int,
        default=None,
        help="Number of speakers in the recording",
    )

    # For now only 2 streams are supported

    args = parser.parse_args()
    assert args.max_neighbors > 1

    # segments file with x-vector timing information. store these in dictionaries
    # indexed by file name
    segs_dict = read_xvector_timing_dict(args.segments_file)

    # Open ark file with x-vectors and in each iteration of the following for-loop
    # read a batch of x-vectors corresponding to one recording
    arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file)

    recit = itertools.groupby(
        arkit, lambda e: e[0].rsplit("_", 2)[0]
    )  # group xvectors in ark by recording name

    for file_name, segs in recit:
        print(file_name)
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        with h5py.File(args.xvec_transform, "r") as f:
            mean1 = np.array(f["mean1"])
            mean2 = np.array(f["mean2"])
            lda = np.array(f["lda"])
            x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

        # Compute pairwise similarity matrix
        if args.plda_file is not None:
            # compute PLDA affinity matrix
            kaldi_plda = read_plda(args.plda_file)
            scr_mx = kaldi_ivector_plda_scoring_dense(kaldi_plda, x)
        else:
            scr_mx = cos_similarity(x)

        labels = NME_SpectralClustering(
            scr_mx,
            None,  # no overlap assignment
            pmin=args.min_neighbors,
            pmax=args.max_neighbors,
            num_clusters=args.num_speakers,
        )

        # Create list of overlapping subsegments
        subsegments = []
        for seg_name, time, label in zip(
            segs_dict[file_name][0], segs_dict[file_name][1], labels
        ):
            channel = seg_name.rsplit("_", 2)[1]
            subsegments.append(Segment(channel, *time, label))

        # We sort the subsegments first by channel and then start time so that all segments of
        # 1 channel are in sequential order followed by all segments of 2 channel and so on.
        subsegments = sorted(subsegments, key=lambda x: (x.channel, x.start))

        # At this point the subsegments are overlapping, since we got them from a
        # sliding window diarization method. We make them contiguous here.
        contiguous_segs = []
        for i, seg in enumerate(subsegments):
            # If it is last segment in channel or last contiguous segment, add it to new_segs
            if (
                i == len(subsegments) - 1
                or seg.channel != subsegments[i + 1].channel
                or seg.end <= subsegments[i + 1].start
            ):
                contiguous_segs.append(
                    Segment(seg.channel, seg.start, seg.end, seg.label)
                )
            # Otherwise split overlapping interval between current and next segment
            else:
                avg = (subsegments[i + 1].start + seg.end) / 2
                contiguous_segs.append(Segment(seg.channel, seg.start, avg, seg.label))
                subsegments[i + 1] = subsegments[i + 1]._replace(start=avg)

        # At this point, we have contiguous subsegments for each channel. But the
        # speaker segments across channels may still overlap. We merge cross-channel
        # segments of the same speaker now.
        segs_by_speaker = {
            spk: list(segs)
            for spk, segs in itertools.groupby(
                sorted(contiguous_segs, key=lambda x: (x.label, x.start)),
                lambda x: x.label,
            )
        }
        merged_segs = []
        for spk, segs in segs_by_speaker.items():
            merged_segs.append(segs[0])
            for i, seg in enumerate(segs[1:]):
                # If start time is before the end time of previous segment, then combine segments
                if float(seg.start) <= float(merged_segs[-1].end):
                    merged_segs[-1] = merged_segs[-1]._replace(end=seg.end)
                else:
                    merged_segs.append(seg)

        mkdir_p(args.out_rttm_dir)
        rttm_str = "SPEAKER {file} 1 {start:.03f} {duration:.03f} <NA> <NA> {label} <NA> <NA>\n"
        with open(os.path.join(args.out_rttm_dir, f"{file_name}.rttm"), "w") as fp:
            for seg in merged_segs:
                duration = seg.end - seg.start
                fp.write(
                    "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>\n".format(
                        file_name, seg.start, duration, seg.label
                    )
                )
