"""Contextual bandit selectors (non-stationary), arms = algorithms.

Planned modules next to this file: ``nonstationary_linucb.py``,
``flow_corrected_ts.py`` (Naghdi & Baheri), ``clustered_bandit.py``
(Dzhoha & Rozora), ``censored_bandit.py`` (Tornede et al., reuses
``asf.preprocessing.schmee_hahn_imputer.schmee_hahn_impute``).
"""

from __future__ import annotations

from asf.online.base import AbstractOnlineSelector


class AbstractContextualBandit(AbstractOnlineSelector):
    """
    Base class for bandit selectors.

    Rewards are negative (penalised) runtimes of the pulled arm. Subclasses
    should only update the pulled arm(s) in ``_observe``; ``_fit_initial`` can
    warm-start the arm models from the fully observed prefix.

    TODO: shared reward construction and exploration bookkeeping.
    """
