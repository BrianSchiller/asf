"""Contextual, non-stationary bandit selectors (arms = algorithms).

Every preset accepts the components of :class:`AbstractContextualBandit`, so
variants are compositions, e.g.::

    LinTSSelector(weighting=ExponentialDiscount(0.99))          # discounted LinTS
    ThompsonSamplingSelector(
        weighting=ExponentialDiscount(0.99) * NearestNeighbours(50)
    )                                                           # DTS-OC-style
    LinTSSelector(confidence=GaussianConfidence())              # fixed confidence
    LinTSSelector(weighting=ConfidenceWeighting(model))         # re-estimated

Presets: UCB, TS, LinUCB, LinTS; DS-UCB, SW-UCB, DS-TS, SW-TS, f-dsw TS
(non-stationary); DTS-OC (clustered). ``imputation=`` (Tornede et al.)
replaces the PAR-k penalty of censored runs by an imputed reward.
"""

from asf.online.selectors.bandits.base import AbstractContextualBandit
from asf.online.selectors.bandits.confidence import ConfidenceModel, GaussianConfidence
from asf.online.selectors.bandits.partitioning import (
    OnlineKMeans,
    Partitioner,
    SinglePartition,
)
from asf.online.selectors.bandits.policies import (
    ExplorationPolicy,
    ThompsonPolicy,
    UCBPolicy,
)
from asf.online.selectors.bandits.posteriors import (
    BayesianLinearRegression,
    BetaPosterior,
    GaussianPosterior,
    Posterior,
)
from asf.online.selectors.bandits.rewards import RewardTransform
from asf.online.selectors.bandits.thompson import (
    DiscountedTSSelector,
    DTSOCSelector,
    FDSWTSSelector,
    LinTSSelector,
    SlidingWindowTSSelector,
    ThompsonSamplingSelector,
)
from asf.online.selectors.bandits.ucb import (
    DiscountedUCBSelector,
    LinUCBSelector,
    SlidingWindowUCBSelector,
    UCBSelector,
)
from asf.online.selectors.bandits.weighting import (
    ConfidenceWeighting,
    ExponentialDiscount,
    LastPulls,
    NearestNeighbours,
    PullDiscount,
    ProductWeighting,
    SlidingWindow,
    Uniform,
    Weighting,
)

__all__ = [
    "AbstractContextualBandit",
    "UCBSelector",
    "LinUCBSelector",
    "ThompsonSamplingSelector",
    "LinTSSelector",
    "DiscountedTSSelector",
    "SlidingWindowTSSelector",
    "FDSWTSSelector",
    "DiscountedUCBSelector",
    "SlidingWindowUCBSelector",
    "DTSOCSelector",
    "Posterior",
    "GaussianPosterior",
    "BayesianLinearRegression",
    "BetaPosterior",
    "ExplorationPolicy",
    "UCBPolicy",
    "ThompsonPolicy",
    "Weighting",
    "Uniform",
    "ExponentialDiscount",
    "PullDiscount",
    "SlidingWindow",
    "LastPulls",
    "NearestNeighbours",
    "ConfidenceWeighting",
    "ProductWeighting",
    "ConfidenceModel",
    "GaussianConfidence",
    "Partitioner",
    "SinglePartition",
    "OnlineKMeans",
    "RewardTransform",
]
