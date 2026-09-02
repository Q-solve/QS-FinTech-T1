"""QKash remittance optimization package."""

# Imports.
from .data import (
    FilterSpec,
    filter_dataset,
    latest_service_options,
    load_dataset,
    match_transfer_amount,
    prepare_candidates,
)
from .profiling import infer_use_case_policy
from .scoring import (
    add_objective_losses,
    build_selection_qubo,
    pareto_prune,
    score_candidates,
    select_qubo_candidates,
    verify_qubo_equivalence,
)

# Variable descriptions.
# __all__ defines the public package surface imported by downstream scripts.
__all__ = [
    "FilterSpec",
    "add_objective_losses",
    "build_selection_qubo",
    "filter_dataset",
    "infer_use_case_policy",
    "latest_service_options",
    "load_dataset",
    "match_transfer_amount",
    "pareto_prune",
    "prepare_candidates",
    "score_candidates",
    "select_qubo_candidates",
    "verify_qubo_equivalence",
]

# __version__ tracks the local starter package version.
__version__ = "0.1.0"
