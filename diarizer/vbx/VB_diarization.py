#!/usr/bin/env python

# Copyright 2013-2019 Lukas Burget, Mireia Diez (burget@fit.vutbr.cz, mireia@fit.vutbr.cz)
#           2022 Desh Raj
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

#
# Revision History
#   16/07/13 01:00AM - original version
#   20/06/17 12:07AM - np.asarray replaced by .toarray()
#                    - minor bug fix in initializing q(Z)
#                    - minor bug fix in ELBO calculation
#                    - few more optimizations
#   03/10/19 02:27PM - speaker regularization coefficient Fb added
#   20/03/20 02:15PM - GMM simplification
#   13/02/22 01:15PM - coupled VBx for cross-stream clustering

import numpy as np
import scipy.linalg as spl


#  [gamma pi Li] =
def VB_diarization(
    X,
    m,
    iE,
    V,
    pi=None,
    gamma=None,
    maxSpeakers=10,
    maxIters=10,
    epsilon=1e-4,
    loopProb=0.99,
    alphaQInit=1.0,
    ref=None,
    plot=False,
    minDur=1,
    Fa=1.0,
    Fb=1.0,
):

    """
    This is a simplified version of speaker diarization described in:

    Diez. M., Burget. L., Landini. F., Cernocky. J.
    Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors

    Variable names and equation numbers refer to those used in the paper

    Inputs:
    X           - T x D array, where columns are D dimensional feature vectors for T frames
    m           - C x D array of GMM component means
    iE          - C x D array of GMM component inverse covariance matrix diagonals
    V           - R x C x D array of eigenvoices
    pi          - speaker priors, if any used for initialization
    gamma       - frame posteriors, if any used for initialization
    maxSpeakers - maximum number of speakers expected in the utterance
    maxIters    - maximum number of algorithm iterations
    epsilon     - stop iterating, if obj. fun. improvement is less than epsilon
    loopProb    - probability of not switching speakers between frames
    alphaQInit  - Dirichlet concentration parameter for initializing gamma
    ref         - T dim. integer vector with reference speaker ID (0:maxSpeakers)
                  per frame
    plot        - if set to True, plot per-frame speaker posteriors.
    minDur      - minimum number of frames between speaker turns imposed by linear
                  chains of HMM states corresponding to each speaker. All the states
                  in a chain share the same output distribution
    Fa          - scale sufficient statistics collected using UBM
    Fb          - speaker regularization coefficient Fb (controls final # of speaker)

    Outputs:
    gamma  - S x T matrix of posteriors attribution each frame to one of S possible
             speakers, where S is given by opts.maxSpeakers
    pi - S dimensional column vector of ML learned speaker priors. Ideally, these
         should allow to estimate # of speaker in the utterance as the
         probabilities of the redundant speaker should converge to zero.
    Li - values of auxiliary function (and DER and frame cross-entropy between gamma
         and reference if 'ref' is provided) over iterations.
    """

    nframes, D = X.shape  # feature dimensionality
    R = V.shape[0]  # subspace rank

    if pi is None:
        pi = np.ones(maxSpeakers) / maxSpeakers
    else:
        maxSpeakers = len(pi)

    if gamma is None:
        # initialize gamma from flat Dirichlet prior with concentration parameter alphaQInit
        gamma = np.random.gamma(alphaQInit, size=(nframes, maxSpeakers))
        gamma = gamma / gamma.sum(1, keepdims=True)

    # calculate UBM mixture frame posteriors (i.e. per-frame zero order statistics)
    G = -0.5 * (
        np.sum((X - m).dot(iE) * (X - m), axis=1) - logdet(iE) + D * np.log(2 * np.pi)
    )
    LL = np.sum(G)  # total log-likelihood as calculated using UBM
    VtiEV = V.dot(iE).dot(V.T)
    VtiEF = (X - m).dot(iE.dot(V).T)

    Li = [[LL * Fa]]  # for the 0-th iteration,
    if ref is not None:
        Li[-1] += [DER(gamma, ref), DER(gamma, ref, xentropy=True)]

    tr = np.eye(minDur * maxSpeakers, k=1)
    ip = np.zeros(minDur * maxSpeakers)
    for i in range(maxIters):
        L = 0  # objective function (37) (i.e. VB lower-bound on the evidence)
        Ns = np.sum(gamma, axis=0)[
            :, np.newaxis, np.newaxis
        ]  # bracket in eq. (34) for all 's'
        VtiEFs = gamma.T.dot(VtiEF)[
            :, :, np.newaxis
        ]  # eq. (35) except for \Lambda_s^{-1} for all 's'
        invLs = np.linalg.inv(
            np.eye(R)[np.newaxis, :, :] + Ns * VtiEV[np.newaxis, :, :] * Fa / Fb
        )  # eq. (34) inverse
        a = np.matmul(invLs, VtiEFs).squeeze(axis=-1) * Fa / Fb  # eq. (35)
        # eq. (29) except for the prior term \ln \pi_s. Our prior is given by HMM
        # transition probability matrix. Instead of eq. (30), we need to use
        # forward-backward algorithm to calculate per-frame speaker posteriors,
        # where 'lls' plays role of HMM output log-probabilities
        lls = Fa * (
            G[:, np.newaxis]
            + VtiEF.dot(a.T)
            - 0.5
            * (
                (invLs + np.matmul(a[:, :, np.newaxis], a[:, np.newaxis, :]))
                * VtiEV[np.newaxis, :, :]
            ).sum(axis=(1, 2))
        )

        for sid in range(maxSpeakers):
            L += (
                Fb
                * 0.5
                * (
                    logdet(invLs[sid])
                    - np.sum(np.diag(invLs[sid]) + a[sid] ** 2, 0)
                    + R
                )
            )

        # Construct transition probability matrix with linear chain of 'minDur'
        # states for each of 'maxSpeaker' speaker. The last state in each chain has
        # self-loop probability 'loopProb' and the transition probabilities to the
        # initial chain states given by vector '(1-loopProb) * pi'. From all other,
        # states, one must move to the next state in the chain with probability one.
        tr[minDur - 1 :: minDur, 0::minDur] = (1 - loopProb) * pi
        tr[(np.arange(1, maxSpeakers + 1) * minDur - 1,) * 2] += loopProb
        ip[::minDur] = pi
        # per-frame HMM state posteriors. Note that we can have linear chain of minDur states
        # for each speaker.
        gamma, tll, lf, lb = forward_backward(lls.repeat(minDur, axis=1), tr, ip)

        # Right after updating q(Z), tll is E{log p(X|,Y,Z)} - KL{q(Z)||p(Z)}.
        # L now contains -KL{q(Y)||p(Y)}. Therefore, L+ttl is correct value for ELBO.
        L += tll
        Li.append([L])

        # ML estimate of speaker prior probabilities (analogue to eq. (38))
        with np.errstate(
            divide="ignore"
        ):  # too close to 0 values do not change the result
            pi = gamma[0, ::minDur] + np.exp(
                logsumexp(lf[:-1, minDur - 1 :: minDur], axis=1)[:, np.newaxis]
                + lb[1:, ::minDur]
                + lls[1:]
                + np.log((1 - loopProb) * pi)
                - tll
            ).sum(axis=0)
        pi = pi / pi.sum()

        # per-frame speaker posteriors (analogue to eq. (30)), obtained by summing
        # HMM state posteriors corresponding to each speaker
        gamma = gamma.reshape(len(gamma), maxSpeakers, minDur).sum(axis=2)

        # if reference is provided, report DER, cross-entropy and plot the figures
        if ref is not None:
            Li[-1] += [DER(gamma, ref), DER(gamma, ref, xentropy=True)]

            if plot:
                import matplotlib.pyplot

                if i == 0:
                    matplotlib.pyplot.clf()
                matplotlib.pyplot.subplot(maxIters, 1, i + 1)
                matplotlib.pyplot.plot(gamma, lw=2)
                matplotlib.pyplot.imshow(
                    np.atleast_2d(ref),
                    interpolation="none",
                    aspect="auto",
                    cmap=matplotlib.pyplot.cm.Pastel1,
                    extent=(0, len(ref), -0.05, 1.05),
                )
            print(i, Li[-2])

        if i > 0 and L - Li[-2][0] < epsilon:
            if L - Li[-1][0] < 0:
                print("WARNING: Value of auxiliary function has decreased!")
            break

    return gamma, pi, Li


# The following is similar to VB_diarization, but is intended to perform clustering across
# K aligned streams of x-vectors, such that each state representes a speaker tuple instead
# of a single speaker.
def VB_diarization_coupled(
    X,
    m,
    iE,
    V,
    pi=None,
    gamma=None,
    maxSpeakers=10,
    maxIters=10,
    epsilon=1e-4,
    loopProbSame=0.99,
    loopProbDiff=0.1,
    alphaQInit=1.0,
    Fa=1.0,
    Fb=1.0,
):

    """
    Inputs:
    X           - K x T x D array, with K streams, each containing T frames and D dimensional x-vectors
    m           - C x D array of GMM component means
    iE          - C x D array of GMM component inverse covariance matrix diagonals
    V           - R x C x D array of eigenvoices
    pi          - speaker pair priors, if any used for initialization
    gamma       - frame posteriors for K streams, if any used for initialization
    maxSpeakers - maximum number of speakers expected in the utterance
    maxIters    - maximum number of algorithm iterations
    epsilon     - stop iterating, if obj. fun. improvement is less than epsilon
    loopProb    - probability of not switching speakers between frames
    alphaQInit  - Dirichlet concentration parameter for initializing gamma
    Fa          - scale sufficient statistics collected using UBM
    Fb          - speaker regularization coefficient Fb (controls final # of speaker)

    Outputs:
    gamma   - K x S x T matrix of posteriors attribution each frame to one of S possible
             speakers, where S is given by opts.maxSpeakers, for each of the K streams
    pi - S x S matrix vector of ML learned priors for speaker pairs. Ideally, these
         should allow to estimate # of speaker in the utterance as the
         probabilities of the redundant speakers should converge to zero.
    Li - values of auxiliary function (and DER and frame cross-entropy between gamma
         and reference if 'ref' is provided) over iterations.
    """

    K, nframes, D = X.shape  # feature dimensionality
    R = V.shape[0]  # subspace rank

    if pi is None:
        # Initial probabilities are 0.1 for speaker pairs and 1 for single speakers
        pi = np.ones(maxSpeakers ** K) * 0.1
        for i in range(maxSpeakers):
            pi[i * (maxSpeakers + 1)] = 1.0
        pi /= pi.sum()
    else:
        ValueError("Initial speaker pair probabilities not supported")

    if gamma is None:
        # initialize gamma from flat Dirichlet prior with concentration parameter alphaQInit
        gamma = np.random.gamma(alphaQInit, size=(K, nframes, maxSpeakers))
        gamma = gamma / gamma.sum(2, keepdims=True)

    # calculate UBM mixture frame posteriors (i.e. per-frame zero order statistics).
    G = -0.5 * (
        np.sum(
            (X - m).dot(iE) * (X - m),
            axis=2,
        )
        - logdet(iE)
        + D * np.log(2 * np.pi)
    )
    LL = np.sum(G)  # total log-likelihood as calculated using UBM
    VtiEV = V.dot(iE).dot(V.T)
    VtiEF = (X - m).dot(iE.dot(V).T)

    Li = [[LL * Fa]]  # for the 0-th iteration,

    tr = np.eye(maxSpeakers ** K)  # transition matrix
    # Construct transition probability matrix with 'maxSpeaker^2' speaker pairs.
    # Note that here we remove the 'minDur' parameter to keep the code simple.
    # This is the key part of the coupled VBx where it differs from original VBx:
    # - for same-speaker pair states, self-loop probability is 'loopProbSame'
    # - for different-speaker pair states, self-loop probability is 'loopProbDiff'
    # - for same-speaker pair states, the transition probability to other states
    # is given by vector '(1-loopProb) * pi'
    # - for mix-speaker pairs, the transition probability to same-speaker pairs
    # for the speakers in the current pair is given by '(1-loopProb) * 0.5', and
    # 0 for every other state. This encodes the constraint that 2 different speakers
    # cannot start or stop speaking at the same time.
    for i in range(maxSpeakers ** K):
        if i % (maxSpeakers+1) == 0:  # same speaker pair
            tr[i, :] = (1 - loopProbSame) * pi
        else:  # mix speaker pair
            tr[i, i % maxSpeakers] = (1 - loopProbDiff) * 0.5
            tr[i, i // maxSpeakers] = (1 - loopProbDiff) * 0.5
    for i in range(maxSpeakers ** K):
        if i % (maxSpeakers + 1) == 0:  # same speaker pair
            tr[i, i] += loopProbSame
        else:  # mix speaker pair
            tr[i, i] += loopProbDiff
    tr /= tr.sum(axis=1, keepdims=True)
    ip = np.zeros(maxSpeakers ** K)  # initial state probabilities
    
    for i in range(maxIters):
        L = 0  # objective function (37) (i.e. VB lower-bound on the evidence)

        # First we fix q(Z_1, Z_2) and update q(Y). Since Y does not depend on the
        # stream, we will stack the streams and use the whole as a matrix for the
        # update equations.
        Ns = np.sum(gamma.reshape((K * nframes, maxSpeakers)), axis=0)[
            :, np.newaxis, np.newaxis
        ]  # bracket in eq. (34) for all 's'
        VtiEFs = gamma.reshape((K * nframes, maxSpeakers)).T.dot(
            VtiEF.reshape((K * nframes, D))
        )[
            :, :, np.newaxis
        ]  # eq. (35) except for \Lambda_s^{-1} for all 's'
        invLs = np.linalg.inv(
            np.eye(R)[np.newaxis, :, :] + Ns * VtiEV[np.newaxis, :, :] * Fa / Fb
        )  # eq. (34) inverse
        a = np.matmul(invLs, VtiEFs).squeeze(axis=-1) * Fa / Fb  # eq. (35)

        # eq. (29) except for the prior term \ln \pi_s. Our prior is given by HMM
        # transition probability matrix. Instead of eq. (30), we need to use
        # forward-backward algorithm to calculate per-frame speaker posteriors,
        # where 'lls' plays role of HMM output log-probabilities
        lls = Fa * (
            G.reshape((K * nframes))[:, np.newaxis]
            + VtiEF.reshape((K * nframes, D)).dot(a.T)
            - 0.5
            * (
                (invLs + np.matmul(a[:, :, np.newaxis], a[:, np.newaxis, :]))
                * VtiEV[np.newaxis, :, :]
            ).sum(axis=(1, 2))
        ).reshape((K, nframes, maxSpeakers))

        for sid in range(maxSpeakers):
            L += (
                Fb
                * 0.5
                * (
                    logdet(invLs[sid])
                    - np.sum(np.diag(invLs[sid]) + a[sid] ** 2, 0)
                    + R
                )
            )
        ip = pi
        # per-frame HMM state posteriors. Note that we can have linear chain of minDur states
        # for each speaker.
        # lls has shape K x T x S. We need to reshape it to T x S**K where S**K is obtained
        # by the outer sum of all streams.
        lls_new = lls[0]
        for k in range(1, K):
            lls_new = np.stack(
                [np.add.outer(lls_new[t], lls[k, t]) for t in range(nframes)], axis=0
            )
            lls_new = lls_new.reshape((nframes, maxSpeakers ** (k + 1)))
        gamma, tll, lf, lb = forward_backward(lls_new, tr, ip)

        # Right after updating q(Z), tll is E{log p(X|,Y,Z)} - KL{q(Z)||p(Z)}.
        # L now contains -KL{q(Y)||p(Y)}. Therefore, L+ttl is correct value for ELBO.
        L += tll
        Li.append([L])

        # ML estimate of speaker prior probabilities (analogue to eq. (38))
        with np.errstate(
            divide="ignore"
        ):  # too close to 0 values do not change the result
            pi = gamma[0, :] + np.exp(
                logsumexp(lf[:-1, :], axis=1)[:, np.newaxis]
                + lb[1:, :]
                + lls_new[1:]
                + np.log(np.dot(tr, pi))
                - tll
            ).sum(axis=0)
        pi = pi / pi.sum()

        # per-frame speaker posteriors (analogue to eq. (30)) --- for this, we sum over
        # all pairs in which the speaker exists. Currently only implemented for K=2
        gamma_new = np.zeros((K, nframes, maxSpeakers))
        for i in range(maxSpeakers):
            gamma_new[0, :, i] = gamma[:, i * maxSpeakers : (i + 1) * maxSpeakers].sum(
                axis=1
            )
            gamma_new[1, :, i] = gamma[:, i::maxSpeakers].sum(axis=1)
        gamma = gamma_new

        if i > 0 and L - Li[-2][0] < epsilon:
            if L - Li[-1][0] < 0:
                print("WARNING: Value of auxiliary function has decreased!")
            break

    return gamma, pi, Li


# Calculates Diarization Error Rate (DER) or per-frame cross-entropy between
# reference (vector of per-frame zero based integer speaker IDs) and gamma (per-frame
# speaker posteriors). If expected=False, gamma is converted into hard labels before
# calculating DER. If expected=TRUE, posteriors in gamma are used to calculated
# "expected" DER.
def DER(gamma, ref, expected=True, xentropy=False):
    from itertools import permutations

    if not expected:
        # replace probabilities in gamma by zeros and ones
        hard_labels = gamma.argmax(1)
        gamma = np.zeros_like(gamma)
        gamma[range(len(gamma)), hard_labels] = 1

    err_mx = np.empty((ref.max() + 1, gamma.shape[1]))
    for s in range(err_mx.shape[0]):
        tmpq = gamma[ref == s, :]
        err_mx[s] = (-np.log(tmpq) if xentropy else tmpq).sum(0)

    if err_mx.shape[0] < err_mx.shape[1]:
        err_mx = err_mx.T

    # try all alignments (permutations) of reference and detected speaker
    # could be written in more efficient way using dynamic programing
    acc = [
        err_mx[perm[: err_mx.shape[1]], range(err_mx.shape[1])].sum()
        for perm in permutations(range(err_mx.shape[0]))
    ]
    if xentropy:
        return min(acc) / float(len(ref))
    return (len(ref) - max(acc)) / float(len(ref))


###############################################################################
# Module private functions
###############################################################################
def logsumexp(x, axis=0):
    xmax = x.max(axis)
    with np.errstate(invalid="ignore"):  # nans do not affect inf
        x = xmax + np.log(np.sum(np.exp(x - np.expand_dims(xmax, axis)), axis))
    infs = np.isinf(xmax)
    if np.ndim(x) > 0:
        x[infs] = xmax[infs]
    elif infs:
        x = xmax
    return x


def logdet(A):
    return 2 * np.sum(np.log(np.diag(spl.cholesky(A))))


def forward_backward(lls, tr, ip):
    """
    Inputs:
        lls - matrix of per-frame log HMM state output probabilities
        tr  - transition probability matrix
        ip  - vector of initial state probabilities (i.e. statrting in the state)
    Outputs:
        pi  - matrix of per-frame state occupation posteriors
        tll - total (forward) log-likelihood
        lfw - log forward probabilities
        lfw - log backward probabilities
    """
    with np.errstate(divide="ignore"):  # too close to 0 values do not change the result
        ltr = np.log(tr)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls)
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    with np.errstate(divide="ignore"):  # too close to 0 values do not change the result
        lfw[0] = lls[0] + np.log(ip)
    lbw[-1] = 0.0

    for i in range(1, len(lls)):
        lfw[i] = lls[i] + logsumexp(lfw[i - 1] + ltr.T, axis=1)

    for i in reversed(range(len(lls) - 1)):
        lbw[i] = logsumexp(ltr + lls[i + 1] + lbw[i + 1], axis=1)

    tll = logsumexp(lfw[-1])
    pi = np.exp(lfw + lbw - tll)
    return pi, tll, lfw, lbw
