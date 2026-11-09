"""An offline selector on a covariate-shifting stream, without adaptation.

Only the distribution of instance features changes. The relationship between
features and algorithm runtimes is identical before and after the shift.
"""

from sklearn.ensemble import RandomForestClassifier

from asf.online.evaluation import run_prequential
from asf.online.selectors import StaticOnlineSelector
from asf.online.streams import make_synthetic_scenario
from asf.selectors import PairwiseClassifier

stream = make_synthetic_scenario(
    "sudden_covariate",
    n_instances=400,
    n_features=3,
    n_algorithms=3,
    breakpoints=[250],
    random_state=0,
)
selector = StaticOnlineSelector(
    PairwiseClassifier(model_class=RandomForestClassifier, random_state=0)
)
result = run_prequential(selector, stream, n_warmstart=100)

for key, value in result.summary().items():
    print(f"{key:>26}: {value:.3f}")
