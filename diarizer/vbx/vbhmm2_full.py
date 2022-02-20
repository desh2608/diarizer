#!/usr/bin/env python

# @Authors: Lukas Burget, Mireia Diez, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, mireia@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com
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


# This script also applies VBx on CSS output streams, but it is a more general version.
# Both the streams are now jointly modeled using a single HMM where the states
# correspond to speaker pairs instead of single speakers. The emissions are still modeled
# by single speaker models.

# An important requirement here is that the subsegments must be aligned in time, i.e.,
# we should make sure their start and end times are always multiples of 0.75 seconds,
# since that is the stride we use for x-vector extraction.

import argparse
import os
import sys
import itertools
from collections import namedtuple

import fastcluster
import h5py
import kaldi_io
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.special import softmax
from scipy.linalg import eigh

from diarizer.diarization_lib import (
    read_xvector_timing_dict,
    l2_norm,
    cos_similarity,
    twoGMMcalib_lin,
    mkdir_p,
)
from diarizer.kaldi_utils import read_plda
from diarizer.vbx.VB_diarization import VB_diarization, VB_diarization_coupled

Segment = namedtuple("Segment", ["channel", "start", "end", "xvec", "label"])
Region = namedtuple("Region", ["start", "end", "xvec_ch0", "xvec_ch1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init",
        required=True,
        type=str,
        choices=["AHC", "AHC+VB", "random_5"],
        help="AHC for using only AHC or AHC+VB for VB-HMM after AHC initilization or random_5 "
        "for running 5 random initializations for VBx and picking the best per-ELBO",
    )
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
        required=True,
        type=str,
        help="File with PLDA model in Kaldi format used for AHC and VB-HMM x-vector clustering",
    )
    parser.add_argument(
        "--threshold",
        required=True,
        type=float,
        help="args.threshold (bias) used for AHC",
    )
    parser.add_argument(
        "--lda-dim",
        required=True,
        type=int,
        help="For VB-HMM, x-vectors are reduced to this dimensionality using LDA",
    )
    parser.add_argument(
        "--Fa",
        required=True,
        type=float,
        help="Parameter of VB-HMM (see VB_diarization.VB_diarization)",
    )
    parser.add_argument(
        "--Fb",
        required=True,
        type=float,
        help="Parameter of VB-HMM (see VB_diarization.VB_diarization)",
    )
    parser.add_argument(
        "--loopP-same",
        required=True,
        type=float,
        help="Parameter of VB-HMM (see VB_diarization.VB_diarization)",
    )
    parser.add_argument(
        "--loopP-diff",
        required=True,
        type=float,
        help="Parameter of VB-HMM (see VB_diarization.VB_diarization)",
    )
    parser.add_argument(
        "--init-smoothing",
        required=False,
        type=float,
        default=10.0,
        help='AHC produces hard assignments of x-vetors to speakers. These are "smoothed" to soft '
        "assignments as the args.initialization for VB-HMM. This parameter controls the amount of"
        " smoothing. Not so important, high value (e.g. 10) is OK  => keeping hard assigment",
    )

    args = parser.parse_args()
    assert (
        0 <= args.loopP_same <= 1
    ), f"Expecting loopP-same between 0 and 1, got {args.loopP_same} instead."
    assert (
        0 <= args.loopP_diff <= 1
    ), f"Expecting loopP-diff between 0 and 1, got {args.loopP_diff} instead."

    # segments file with x-vector timing information
    segs_dict = read_xvector_timing_dict(args.segments_file)

    kaldi_plda = read_plda(args.plda_file)
    plda_mu, plda_tr, plda_psi = kaldi_plda
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    # Open ark file with x-vectors and in each iteration of the following for-loop
    # read a batch of x-vectors corresponding to one recording
    arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file)
    recit = itertools.groupby(
        arkit, lambda e: e[0].rsplit("_", 2)[0]
    )  # group xvectors in ark by recording name
    for file_name, segs in recit:
        print(file_name)
        seg_names, xvecs = zip(*segs)
        starts, ends = segs_dict[file_name][1].T

        # Create subsegments from the xvectors (just assign dummy labels for now)
        subsegments = []
        for seg_name, xvec, start, end in zip(seg_names, xvecs, starts, ends):
            subsegments.append(
                Segment(seg_name.rsplit("_", 2)[1], start, end, xvec, -1)
            )
        # Group subsegments by start time
        subsegments = sorted(subsegments, key=lambda e: e.start)
        subsegments = itertools.groupby(subsegments, key=lambda e: e.start)
        regions = []
        # We create regions from subsegments, where each region contains x-vector from both
        # streams. If only one stream is available, we just duplicate the x-vector.
        for start, segs in subsegments:
            segs = list(segs)
            if len(segs) == 1:
                regions.append(Region(start, segs[0].end, segs[0].xvec, segs[0].xvec))
            else:
                regions.append(Region(start, segs[0].end, segs[0].xvec, segs[-1].xvec))
        x0 = np.array([r.xvec_ch0 for r in regions])  # x-vectors from stream 0
        x1 = np.array([r.xvec_ch1 for r in regions])  # x-vectors from stream 1
        x = np.concatenate((x0, x1), axis=0)

        with h5py.File(args.xvec_transform, "r") as f:
            mean1 = np.array(f["mean1"])
            mean2 = np.array(f["mean2"])
            lda = np.array(f["lda"])
            x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

        if (
            args.init == "AHC"
            or args.init.endswith("VB")
            or args.init.startswith("random_")
        ):
            # if args.init.startswith("AHC"):
            #     # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise
            #     # similarities between all x-vectors)
            #     scr_mx = cos_similarity(x)
            #     # Figure out utterance specific args.threshold for AHC.
            #     thr, _ = twoGMMcalib_lin(scr_mx.ravel())
            #     # output "labels" is an integer vector of speaker (cluster) ids
            #     scr_mx = squareform(-scr_mx, checks=False)
            #     lin_mat = fastcluster.linkage(
            #         scr_mx, method="average", preserve_input="False"
            #     )
            #     del scr_mx
            #     adjust = abs(lin_mat[:, 2].min())
            #     lin_mat[:, 2] += adjust
            #     labels = (
            #         fcluster(
            #             lin_mat, -(thr + args.threshold) + adjust, criterion="distance"
            #         )
            #         - 1
            #     )
            #     with open("OV20_session0.npy", "wb") as f:
            #         np.save(f, labels)
            #     sys.exit(1)
            if args.init.endswith("VB"):
                with open("OV20_session0.npy", "rb") as f:
                    labels = np.load(f)
                labels0 = labels[: len(x0)]
                labels1 = labels[len(x0) :]
                assert len(labels0) == len(labels1)
                # Now we create a Q-matrix with shape (2, T, K).
                T = len(labels0)
                K = np.max(labels) + 1
                qinit = np.zeros((2, T, K))
                qinit[0, range(T), labels0] = 1.0
                qinit[1, range(T), labels1] = 1.0
                qinit = softmax(qinit * args.init_smoothing, axis=2)

                fea = (x - plda_mu).dot(plda_tr.T)[:, : args.lda_dim]
                # Use VB-HMM for x-vector clustering. Instead of i-vector extractor model, we use PLDA
                # => GMM with only 1 component, V derived accross-class covariance,
                # and iE is inverse within-class covariance (i.e. identity)
                sm = np.zeros(args.lda_dim)
                siE = np.ones(args.lda_dim)
                sV = np.sqrt(plda_psi[: args.lda_dim])
                # Divide fea into the 2 channel inputs
                fea0 = fea[: len(x0)]
                fea1 = fea[len(x0) :]
                fea = np.stack((fea0, fea1), axis=0)
                q, sp, L = VB_diarization_coupled(
                    fea,
                    sm,
                    np.diag(siE),
                    np.diag(sV),
                    pi=None,
                    gamma=qinit,
                    maxSpeakers=K,
                    maxIters=40,
                    epsilon=1e-6,
                    loopProbSame=args.loopP_same,
                    loopProbDiff=args.loopP_diff,
                    Fa=args.Fa,
                    Fb=args.Fb,
                )

                labels0 = np.argsort(-q[0], axis=1)[:, 0]
                labels1 = np.argsort(-q[1], axis=1)[:, 0]
            if args.init.startswith("random_"):
                MAX_SPKS = 10
                prev_L = -float("inf")
                random_iterations = int(args.init.split("_")[1])
                np.random.seed(3)  # for reproducibility
                for _ in range(random_iterations):
                    q_init = np.random.normal(
                        size=(2, x.shape[0], MAX_SPKS), loc=0.5, scale=0.01
                    )
                    q_init = softmax(q_init * args.init_smoothing, axis=2)
                    fea = (x - plda_mu).dot(plda_tr.T)[:, : args.lda_dim]
                    sm = np.zeros(args.lda_dim)
                    siE = np.ones(args.lda_dim)
                    sV = np.sqrt(plda_psi[: args.lda_dim])
                    q_tmp, sp, L = VB_diarization(
                        fea,
                        sm,
                        np.diag(siE),
                        np.diag(sV),
                        pi=None,
                        gamma=q_init,
                        maxSpeakers=q_init.shape[1],
                        maxIters=40,
                        epsilon=1e-6,
                        loopProb=args.loopP,
                        Fa=args.Fa,
                        Fb=args.Fb,
                    )
                    if L[-1][0] > prev_L:
                        prev_L = L[-1][0]
                        q = q_tmp
                labels0 = np.argsort(-q[0], axis=1)[:, 0]
                labels1 = np.argsort(-q[1], axis=1)[:, 0]
        else:
            raise ValueError("Wrong option for args.initialization.")

        assert np.all(segs_dict[file_name][0] == np.array(seg_names))
        start, end = segs_dict[file_name][1].T

        # split the labels into two channels and process both channels
        subsegments = []
        for i, region in enumerate(regions):
            subsegments.append(Segment(0, region.start, region.end, None, labels0[i]))
            subsegments.append(Segment(1, region.start, region.end, None, labels1[i]))

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
                    Segment(seg.channel, seg.start, seg.end, None, seg.label)
                )
            # Otherwise split overlapping interval between current and next segment
            else:
                avg = (subsegments[i + 1].start + seg.end) / 2
                contiguous_segs.append(Segment(seg.channel, seg.start, avg, None, seg.label))
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
