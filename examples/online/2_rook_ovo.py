"""Online pairwise learning under recurring covariate shift.

This cold-start example compares a frozen pairwise selector with periodic
retraining. Feature distributions alternate A-B-A while the algorithm runtime
model remains identical at every point. Each proposal runs two algorithms;
user cost and total compute cost must both be considered.
"""

from asf.online.evaluation import run_prequential
from asf.online.retraining import PeriodicRetrain
from asf.online.selectors import OnlinePairwiseSelector
from asf.online.streams import make_synthetic_scenario

stream = make_synthetic_scenario(
    "recurring_covariate",
    n_instances=600,
    breakpoints=[300, 450],
    n_features=3,
    n_algorithms=3,
    random_state=0,
)

for name, policy in [
    ("never retrain", None),
    ("retrain every 25", PeriodicRetrain(25)),
]:
    selector = OnlinePairwiseSelector(
        reference_period=50, retrain_policy=policy, random_state=0
    )
    summary = run_prequential(selector, stream).summary()
    print(
        f"{name:>18}: user cost {summary['mean_user_cost']:6.2f}, "
        f"compute cost {summary['mean_compute_cost']:6.2f} "
        f"(SBS {summary['sbs_cost']:.1f}, VBS {summary['vbs_cost']:.1f}), "
        f"closed gap {summary['closed_gap']:.2f}, "
        f"retrains {int(summary['n_retrains'])}, "
        f"retrain time {summary['retrain_seconds']:.2f}s"
    )
