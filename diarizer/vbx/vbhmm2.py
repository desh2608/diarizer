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


# This is similar to vbhmm.py, but it applies VBx on CSS separated output streams.
# The idea is that the HMM transitions on both streams are independent, but the
# speaker states are shared. This is equivalent to saying that the 2 streams represent
# 2 different sessions but with the same set of speakers.

import argparse
import os
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
from diarizer.vbx.VB_diarization import VB_diarization

Segment = namedtuple("Segment", ["channel", "start", "end", "label"])


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
        "--loopP",
        required=True,
        type=float,
        help="Parameter of VB-HMM (see VB_diarization.VB_diarization)",
    )
    parser.add_argument(
        "--init-smoothing",
        required=False,
        type=float,
        default=5.0,
        help='AHC produces hard assignments of x-vetors to speakers. These are "smoothed" to soft '
        "assignments as the args.initialization for VB-HMM. This parameter controls the amount of"
        " smoothing. Not so important, high value (e.g. 10) is OK  => keeping hard assigment",
    )

    args = parser.parse_args()
    assert (
        0 <= args.loopP <= 1
    ), f"Expecting loopP between 0 and 1, got {args.loopP} instead."

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
        # sort segs by name (so that all channel 0 are first)
        segs = sorted(segs, key=lambda e: e[0])
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        # get number of segments in each channel (will be used later to split the labels)
        num_ch0 = len(list(filter(lambda e: e.rsplit("_", 2)[1] == "0", seg_names)))
        num_ch1 = len(list(filter(lambda e: e.rsplit("_", 2)[1] == "1", seg_names)))

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
            if args.init.startswith("AHC"):
                # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise
                # similarities between all x-vectors)
                scr_mx = cos_similarity(x)
                # Figure out utterance specific args.threshold for AHC.
                thr, _ = twoGMMcalib_lin(scr_mx.ravel())
                # output "labels" is an integer vector of speaker (cluster) ids
                scr_mx = squareform(-scr_mx, checks=False)
                lin_mat = fastcluster.linkage(
                    scr_mx, method="average", preserve_input="False"
                )
                del scr_mx
                adjust = abs(lin_mat[:, 2].min())
                lin_mat[:, 2] += adjust
                labels = (
                    fcluster(
                        lin_mat, -(thr + args.threshold) + adjust, criterion="distance"
                    )
                    - 1
                )
            if args.init.endswith("VB"):
                # Smooth the hard labels obtained from AHC to soft assignments
                # of x-vectors to speakers. At this point, the Q-matrix will contain
                # number of rows equal to the number of subsegments in both the channels.
                qinit = np.zeros((len(labels), np.max(labels) + 1))
                qinit[range(len(labels)), labels] = 1.0
                qinit = softmax(qinit * args.init_smoothing, axis=1)
                fea = (x - plda_mu).dot(plda_tr.T)[:, : args.lda_dim]
                # Use VB-HMM for x-vector clustering. Instead of i-vector extractor model, we use PLDA
                # => GMM with only 1 component, V derived accross-class covariance,
                # and iE is inverse within-class covariance (i.e. identity)
                sm = np.zeros(args.lda_dim)
                siE = np.ones(args.lda_dim)
                sV = np.sqrt(plda_psi[: args.lda_dim])
                q, sp, L = VB_diarization(
                    fea,
                    sm,
                    np.diag(siE),
                    np.diag(sV),
                    pi=None,
                    gamma=qinit,
                    maxSpeakers=qinit.shape[1],
                    maxIters=40,
                    epsilon=1e-6,
                    loopProb=args.loopP,
                    Fa=args.Fa,
                    Fb=args.Fb,
                )

                labels = np.argsort(-q, axis=1)[:, 0]
            if args.init.startswith("random_"):
                MAX_SPKS = 10
                prev_L = -float("inf")
                random_iterations = int(args.init.split("_")[1])
                np.random.seed(3)  # for reproducibility
                for _ in range(random_iterations):
                    q_init = np.random.normal(
                        size=(x.shape[0], MAX_SPKS), loc=0.5, scale=0.01
                    )
                    q_init = softmax(q_init * args.init_smoothing, axis=1)
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
                labels = np.argsort(-q, axis=1)[:, 0]
        else:
            raise ValueError("Wrong option for args.initialization.")

        assert np.all(segs_dict[file_name][0] == np.array(seg_names))
        start, end = segs_dict[file_name][1].T

        # split the labels into two channels and process both channels
        labels_ch0 = labels[:num_ch0]
        labels_ch1 = labels[num_ch0:]
        start_ch0 = start[:num_ch0]
        start_ch1 = start[num_ch0:]
        end_ch0 = end[:num_ch0]
        end_ch1 = end[num_ch0:]
        subsegments = []
        for (label, start, end) in zip(labels_ch0, start_ch0, end_ch0):
            subsegments.append(Segment(0, start, end, label))
        for (label, start, end) in zip(labels_ch1, start_ch1, end_ch1):
            subsegments.append(Segment(1, start, end, label))

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