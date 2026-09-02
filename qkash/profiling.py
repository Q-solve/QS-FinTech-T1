"""Service-profile clustering and internal policy inference for QKash."""

from __future__ import annotations

# Imports.
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from .scoring import LOSS_COLUMNS, normalize_weights


# Variable descriptions.
# DEFAULT_PROFILE is used when the candidate set is too small for useful ML.
DEFAULT_PROFILE = "balanced"

# PROFILE_FEATURE_COLUMNS are the numeric inputs used by K-Means and the classifier.
PROFILE_FEATURE_COLUMNS = (
    "transaction_fee_loss",
    "fx_spread_loss",
    "time_loss",
    "cash_indicator",
    "bank_indicator",
    "mobile_digital_indicator",
)

# PROFILE_LABELS maps internal profile ids to consumer-readable names.
PROFILE_LABELS = {
    "low_cost": "Low-cost service",
    "fast_transfer": "Fast-transfer service",
    "balanced": "Balanced service",
    "cash_oriented": "Cash-oriented service",
    "bank_based": "Bank-based service",
    "mobile_digital": "Mobile-digital service",
}

# PROFILE_POLICIES convert the inferred profile into internal objective weights.
PROFILE_POLICIES = {
    "low_cost": {"transaction_fee": 0.50, "fx_spread": 0.35, "time": 0.15},
    "fast_transfer": {"time": 0.60, "transaction_fee": 0.25, "fx_spread": 0.15},
    "balanced": {"transaction_fee": 0.34, "fx_spread": 0.33, "time": 0.33},
    "cash_oriented": {"transaction_fee": 0.40, "time": 0.35, "fx_spread": 0.25},
    "bank_based": {"fx_spread": 0.40, "transaction_fee": 0.35, "time": 0.25},
    "mobile_digital": {"time": 0.45, "transaction_fee": 0.35, "fx_spread": 0.20},
}

# DEFAULT_CLUSTER_COUNT is the maximum number of natural service profiles to seek.
DEFAULT_CLUSTER_COUNT = len(PROFILE_LABELS)

# RANDOM_FOREST_TREES controls classifier stability without making each run heavy.
RANDOM_FOREST_TREES = 160


@dataclass(frozen=True)
class ProfileInference:
    """Internal transaction profile and optimization policy."""

    profile: str
    profile_label: str
    weights: dict[str, float]
    confidence: float
    probabilities: dict[str, float]
    cluster_count: int
    candidates_profiled: int
    note: str


def infer_use_case_policy(
    candidates: pd.DataFrame,
    transfer_amount: float,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, ProfileInference]:
    """Cluster services and infer the transaction's internal policy weights.

    The RPW dataset does not include supervised consumer-use-case labels.  QKash
    therefore first creates natural service groups with K-Means, translates
    those groups into semantic profiles, and trains a Random Forest classifier
    on the cluster-derived labels.  The classifier's average probability over
    the current candidate set is treated as the transaction-context profile.
    """

    profiled = add_profile_features(candidates)
    if profiled.empty:
        return profiled, _default_inference(transfer_amount, "No candidates available.")

    feature_frame = _profile_feature_frame(profiled)
    cluster_count = min(DEFAULT_CLUSTER_COUNT, len(feature_frame))

    if cluster_count <= 1:
        profile = _single_row_profile(feature_frame.iloc[0])
        profiled["service_profile"] = profile
        profiled["service_profile_label"] = PROFILE_LABELS[profile]
        return profiled, _inference_from_profile(
            profile,
            confidence=1.0,
            probabilities={profile: 1.0},
            cluster_count=1,
            candidates_profiled=len(profiled),
            note="Only one service profile was available after Pareto pruning.",
        )

    kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=random_state)
    cluster_ids = kmeans.fit_predict(feature_frame)
    cluster_profiles = _name_clusters(profiled, feature_frame, cluster_ids, kmeans.cluster_centers_)
    profile_labels = [cluster_profiles[int(cluster_id)] for cluster_id in cluster_ids]

    profiled = profiled.copy()
    profiled["cluster_id"] = cluster_ids.astype(int)
    profiled["service_profile"] = profile_labels
    profiled["service_profile_label"] = [
        PROFILE_LABELS.get(profile, PROFILE_LABELS[DEFAULT_PROFILE])
        for profile in profile_labels
    ]

    probabilities = _random_forest_profile_probabilities(
        feature_frame,
        pd.Series(profile_labels, index=feature_frame.index),
        random_state=random_state,
    )
    profile = max(probabilities, key=probabilities.get)
    return profiled, _inference_from_profile(
        profile,
        confidence=float(probabilities[profile]),
        probabilities=probabilities,
        cluster_count=cluster_count,
        candidates_profiled=len(profiled),
        note="K-Means service profiles classified with Random Forest pseudo-labels.",
    )


def add_profile_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add binary method-channel features used to profile services."""

    profiled = candidates.copy()
    if profiled.empty:
        return profiled

    combined_text = (
        _text_column(profiled, "payment instrument")
        + " "
        + _text_column(profiled, "pickup method")
        + " "
        + _text_column(profiled, "firm_type")
    ).str.lower()
    profiled["cash_indicator"] = combined_text.str.contains("cash", regex=False).astype(float)
    profiled["bank_indicator"] = (
        combined_text.str.contains("bank", regex=False)
        | combined_text.str.contains("account", regex=False)
    ).astype(float)
    profiled["mobile_digital_indicator"] = (
        combined_text.str.contains("mobile", regex=False)
        | combined_text.str.contains("wallet", regex=False)
        | combined_text.str.contains("card", regex=False)
        | combined_text.str.contains("online", regex=False)
        | combined_text.str.contains("internet", regex=False)
    ).astype(float)
    return profiled


def profile_weight_table(inference: ProfileInference) -> pd.DataFrame:
    """Return the internally inferred objective weights as a display table."""

    return pd.DataFrame(
        [
            {"objective": objective.replace("_", " ").title(), "weight": weight}
            for objective, weight in inference.weights.items()
        ]
    )


def profile_summary(profiled: pd.DataFrame) -> pd.DataFrame:
    """Summarize discovered service profiles for the research dashboard."""

    if profiled.empty or "service_profile_label" not in profiled.columns:
        return pd.DataFrame()

    grouped = profiled.groupby("service_profile_label", dropna=False)
    return (
        grouped.agg(
            services=("firm", "count"),
            mean_fee_loss=("transaction_fee_loss", "mean"),
            mean_fx_loss=("fx_spread_loss", "mean"),
            mean_time_loss=("time_loss", "mean"),
            mean_score=("weighted_score", "mean")
            if "weighted_score" in profiled.columns
            else ("transaction_fee_loss", "mean"),
        )
        .reset_index()
        .rename(columns={"service_profile_label": "service_profile"})
        .sort_values("services", ascending=False, kind="mergesort")
    )


def _profile_feature_frame(profiled: pd.DataFrame) -> pd.DataFrame:
    """Return a complete numeric feature matrix for profile inference."""

    features = profiled.copy()
    for column in LOSS_COLUMNS:
        if column not in features.columns:
            features[column] = 0.0
    for column in PROFILE_FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    return features.loc[:, PROFILE_FEATURE_COLUMNS].clip(0.0, 1.0)


def _text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a string column or a blank series when the column is absent."""

    if column in frame.columns:
        return frame[column].fillna("").astype(str)
    return pd.Series("", index=frame.index, dtype=str)


def _name_clusters(
    profiled: pd.DataFrame,
    feature_frame: pd.DataFrame,
    cluster_ids: np.ndarray,
    centers: np.ndarray,
) -> dict[int, str]:
    """Translate K-Means cluster centroids into business-readable profiles."""

    cluster_profiles: dict[int, str] = {}
    for cluster_id, center in enumerate(centers):
        rows = profiled.loc[cluster_ids == cluster_id]
        cluster_profiles[cluster_id] = _semantic_profile(
            pd.Series(center, index=feature_frame.columns),
            rows,
        )
    return cluster_profiles


def _semantic_profile(center: pd.Series, rows: pd.DataFrame) -> str:
    """Name a service profile from its cost/time centroid and channel mix."""

    cash_share = float(center.get("cash_indicator", 0.0))
    bank_share = float(center.get("bank_indicator", 0.0))
    mobile_share = float(center.get("mobile_digital_indicator", 0.0))
    fee_loss = float(center.get("transaction_fee_loss", 0.0))
    fx_loss = float(center.get("fx_spread_loss", 0.0))
    time_loss = float(center.get("time_loss", 0.0))

    if mobile_share >= 0.55 and mobile_share >= max(cash_share, bank_share):
        return "mobile_digital"
    if bank_share >= 0.55 and bank_share >= max(cash_share, mobile_share):
        return "bank_based"
    if cash_share >= 0.55 and cash_share >= max(bank_share, mobile_share):
        return "cash_oriented"
    if time_loss <= 0.25 and time_loss <= fee_loss and time_loss <= fx_loss:
        return "fast_transfer"
    if fee_loss <= 0.30 and fx_loss <= 0.35:
        return "low_cost"
    if rows.empty:
        return DEFAULT_PROFILE
    return DEFAULT_PROFILE


def _single_row_profile(row: pd.Series) -> str:
    """Name a profile when K-Means has only one candidate to cluster."""

    return _semantic_profile(row, pd.DataFrame([row]))


def _random_forest_profile_probabilities(
    feature_frame: pd.DataFrame,
    profile_labels: pd.Series,
    random_state: int | None,
) -> dict[str, float]:
    """Train a classifier on K-Means labels and average current predictions."""

    classes = sorted(set(str(label) for label in profile_labels))
    if len(classes) == 1:
        return {classes[0]: 1.0}

    classifier = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_TREES,
        max_depth=6,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=random_state,
    )
    classifier.fit(feature_frame, profile_labels.astype(str))

    class_probabilities = classifier.predict_proba(feature_frame)
    averaged = class_probabilities.mean(axis=0)
    probabilities = {
        str(profile): float(probability)
        for profile, probability in zip(classifier.classes_, averaged)
    }
    total = sum(probabilities.values())
    if total <= 0:
        return {DEFAULT_PROFILE: 1.0}
    return {profile: probability / total for profile, probability in probabilities.items()}


def _inference_from_profile(
    profile: str,
    confidence: float,
    probabilities: dict[str, float],
    cluster_count: int,
    candidates_profiled: int,
    note: str,
) -> ProfileInference:
    """Create a normalized profile inference object."""

    normalized_profile = profile if profile in PROFILE_POLICIES else DEFAULT_PROFILE
    return ProfileInference(
        profile=normalized_profile,
        profile_label=PROFILE_LABELS[normalized_profile],
        weights=normalize_weights(PROFILE_POLICIES[normalized_profile]),
        confidence=float(confidence),
        probabilities=probabilities,
        cluster_count=int(cluster_count),
        candidates_profiled=int(candidates_profiled),
        note=note,
    )


def _default_inference(transfer_amount: float, note: str) -> ProfileInference:
    """Return the balanced fallback profile for empty or unusable candidate sets."""

    return _inference_from_profile(
        DEFAULT_PROFILE,
        confidence=0.0,
        probabilities={DEFAULT_PROFILE: 1.0},
        cluster_count=0,
        candidates_profiled=0,
        note=f"{note} Transfer amount: {float(transfer_amount):.2f}.",
    )
