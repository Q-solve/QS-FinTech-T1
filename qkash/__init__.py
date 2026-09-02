"""QKash remittance optimization package."""

# Imports.
from .data import FilterSpec, filter_dataset, load_dataset, prepare_candidates
from .scoring import (
    build_selection_qubo,
    pareto_prune,
    parse_weight_query,
    score_candidates,
    verify_qubo_equivalence,
)

# Variable descriptions.
# __all__ defines the public package surface imported by downstream scripts.
__all__ = [
    "FilterSpec",
    "build_selection_qubo",
    "filter_dataset",
    "load_dataset",
    "pareto_prune",
    "parse_weight_query",
    "prepare_candidates",
    "score_candidates",
    "verify_qubo_equivalence",
]

# __version__ tracks the local starter package version.
__version__ = "0.1.0"
