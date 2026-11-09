"""Contextual, non-stationary bandit selectors (arms = algorithms).

Baselines: :class:`UCBSelector`, :class:`ThompsonSamplingSelector`,
:class:`LinUCBSelector`, :class:`LinTSSelector`. Each accepts the same
components as :class:`AbstractContextualBandit`, so non-stationary variants are
compositions, e.g.::

    LinTSSelector(weighting=ExponentialDiscount(0.99))          # discounted LinTS
    ThompsonSamplingSelector(
        weighting=ExponentialDiscount(0.99) * NearestNeighbours(50)
    )                                                           # DTS-OC-style
    LinTSSelector(confidence=GaussianConfidence())              # fixed confidence
    LinTSSelector(weighting=ConfidenceWeighting(model))         # re-estimated

Planned extensions: online-clustering / dyadic-tree ``Partitioner`` (DTS-OC),
flow-based ``ConfidenceModel`` and Normal-Inverse-Gamma ``Posterior`` (FC-TS),
censored-runtime imputation via ``_rewards`` (Tornede et al.).
"""

from asf.online.selectors.bandits.base import AbstractContextualBandit
from asf.online.selectors.bandits.confidence import ConfidenceModel, GaussianConfidence
from asf.online.selectors.bandits.context import ContextBuilder, model_input
from asf.online.selectors.bandits.history import ArmData, BanditHistory
from asf.online.selectors.bandits.partitioning import Partitioner, SinglePartition
from asf.online.selectors.bandits.policies import (
    ExplorationPolicy,
    ThompsonPolicy,
    UCBPolicy,
)
from asf.online.selectors.bandits.posteriors import (
    BayesianLinearRegression,
    GaussianPosterior,
    Posterior,
)
from asf.online.selectors.bandits.rewards import RewardTransform
from asf.online.selectors.bandits.thompson import (
    LinTSSelector,
    ThompsonSamplingSelector,
)
from asf.online.selectors.bandits.ucb import LinUCBSelector, UCBSelector
from asf.online.selectors.bandits.weighting import (
    ConfidenceWeighting,
    ExponentialDiscount,
    NearestNeighbours,
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
    "Posterior",
    "GaussianPosterior",
    "BayesianLinearRegression",
    "ExplorationPolicy",
    "UCBPolicy",
    "ThompsonPolicy",
    "Weighting",
    "Uniform",
    "ExponentialDiscount",
    "SlidingWindow",
    "NearestNeighbours",
    "ConfidenceWeighting",
    "ProductWeighting",
    "ConfidenceModel",
    "GaussianConfidence",
    "Partitioner",
    "SinglePartition",
    "RewardTransform",
    "ContextBuilder",
    "model_input",
    "BanditHistory",
    "ArmData",
]
