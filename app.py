"""QKash Streamlit application."""

from __future__ import annotations

# Imports.
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from qkash.benchmark import (
    metrics_frame,
    rank_metrics,
    summarize_samples,
    validate_solver_outputs,
)
from qkash.batch import (
    DEFAULT_BATCH_CANDIDATE_CAP,
    DEFAULT_BATCH_PLAN_CAP,
    BatchSettings,
    build_batch_plans,
)
from qkash.classical import (
    business_as_usual_baseline,
    exact_mathematical_baseline,
    fastest_transfer_heuristic,
    lowest_fee_heuristic,
    lowest_fx_heuristic,
    simulated_annealing,
)
from qkash.constraints import PolicyConstraints, apply_policy_constraints
from qkash.data import (
    DEFAULT_DATA_PATH,
    FilterSpec,
    filter_dataset,
    latest_service_options,
    load_dataset,
    match_transfer_amount,
    prepare_candidates,
    unique_values,
)
from qkash.profiling import (
    infer_use_case_policy,
    profile_summary,
    profile_weight_table,
)
from qkash.qbraid_bridge import qbraid_status
from qkash.quantum import DEFAULT_MAX_QUBITS, run_qaoa
from qkash.scoring import (
    add_objective_losses,
    build_selection_qubo,
    pareto_prune,
    required_qubits,
    score_candidates,
    select_qubo_candidates,
    solve_original_exact,
    verify_qubo_equivalence,
)


# Variable descriptions.
# PAGE_TITLE is the name shown in the browser tab and app title.
PAGE_TITLE = "QKash"

# APP_ROOT is the Streamlit application root folder.
APP_ROOT = Path(__file__).resolve().parent

# LOGO_PATH points to the QKash logo stored with the project data assets.
LOGO_PATH = APP_ROOT / "data" / "QKash.png"

# MODEL_CANDIDATE_CAP limits the QUBO size for local QAOA simulation.
MODEL_CANDIDATE_CAP = DEFAULT_MAX_QUBITS

# LOCAL_AER_DEVICE_ID names the local simulator shown in the UI.
LOCAL_AER_DEVICE_ID = "local-aer-simulator"

# QBRAID_SIMULATOR_DEVICE_ID is the default online qBraid gate-model simulator.
QBRAID_SIMULATOR_DEVICE_ID = "qbraid:qbraid:sim:qir-sv"

# QUANTUM_BACKEND_OPTIONS maps UI labels to backend and device selections.
QUANTUM_BACKEND_OPTIONS = {
    "Local Aer simulator": {
        "backend_name": "local_aer",
        "device_id": LOCAL_AER_DEVICE_ID,
    },
    "qBraid simulator": {
        "backend_name": "qbraid",
        "device_id": QBRAID_SIMULATOR_DEVICE_ID,
    },
}


st.set_page_config(page_title=PAGE_TITLE, page_icon=str(LOGO_PATH), layout="wide")


@st.cache_data(show_spinner=False)
def cached_dataset(path: str, modified_ns: int, size_bytes: int) -> pd.DataFrame:
    """Load the CSV once per path into Streamlit's data cache."""

    return load_dataset(path)


def render_app_header() -> None:
    """Render the QKash logo and title at the top of the main app."""

    if not LOGO_PATH.exists():
        st.title(PAGE_TITLE)
        return

    logo_col, title_col = st.columns([1, 7])
    with logo_col:
        st.image(str(LOGO_PATH), width=92)
    with title_col:
        st.title(PAGE_TITLE)


def required_selectbox(label: str, options: list[str]) -> str:
    """Render a required selectbox and stop if no valid option exists."""

    if not options:
        st.error(f"No available options for {label}.")
        st.stop()
    return st.selectbox(label, options)


def display_candidate(row: pd.Series) -> dict[str, object]:
    """Select the candidate fields shown in solution tables."""

    return {
        "firm": row["firm"],
        "corridor": row["corridor"],
        "period": row["period"],
        "service_profile": row.get("service_profile_label", ""),
        "payment_method": row["payment instrument"],
        "receiving_method": row["pickup method"],
        "settlement_time": row["speed actual"],
        "transaction_fee": row["fee_lcu"],
        "fx_spread": row["fx_margin"],
        "risk_loss": row.get("risk_loss", np.nan),
        "overall_performance": performance_score(row),
        "weighted_score": row["weighted_score"],
    }


def best_index_from_result(result: dict[str, object]) -> int | None:
    """Return the best feasible selected candidate index from a solver result."""

    if result.get("best_index") is not None:
        return int(result["best_index"])

    feasible = [
        sample
        for sample in result.get("samples", [])
        if sample.get("feasible") and sample.get("objective") is not None
    ]
    if not feasible:
        return None
    best = min(feasible, key=lambda sample: float(sample["objective"]))
    return int(best["index"])


def solution_frame(results: list[dict[str, object]], candidates: pd.DataFrame) -> pd.DataFrame:
    """Build the selected-solution table for all solvers."""

    rows: list[dict[str, object]] = []
    for result in results:
        index = best_index_from_result(result)
        if index is None or index >= len(candidates):
            rows.append(
                {
                    "algorithm": result["algorithm"],
                    "status": result.get("status", "no feasible solution"),
                    "note": result.get("note", ""),
                }
            )
            continue

        selected = display_candidate(candidates.iloc[index])
        selected["algorithm"] = result["algorithm"]
        selected["status"] = result.get("status", "ok")
        selected["note"] = result.get("note", "")
        rows.append(selected)

    return pd.DataFrame(rows)


def feasible_solution_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return every feasible decoded service solution ranked by weighted score."""

    ranked = candidates.copy().reset_index(drop=True)
    ranked = ranked.sort_values(
        "weighted_score",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for rank, (_index, row) in enumerate(ranked.iterrows(), start=1):
        rows.append(
            {
                "rank": rank,
                "binary_state": row.get("binary_state", ""),
                "service_provider": row["firm"],
                "payment_method": row["payment instrument"],
                "receiving_method": row["pickup method"],
                "settlement_time": row["speed actual"],
                "transaction_fee": fee_text(row),
                "fx_spread": f"{float(row['fx_margin']):.2f}%",
                "risk_loss": float(row.get("risk_loss", 0.0)),
                "service_profile": row.get("service_profile_label", ""),
                "selected_services": row.get("batch_selected_services", ""),
                "overall_performance": performance_score(row),
                "weighted_score": float(row["weighted_score"]),
            }
        )
    return pd.DataFrame(rows)


def performance_score(row: pd.Series) -> float:
    """Convert the lower-is-better objective score into a 0-100 performance score."""

    score = float(row.get("weighted_score", 1.0))
    return max(0.0, min(100.0, 100.0 * (1.0 - score)))


def performance_band(score: float) -> str:
    """Return a concise consumer label for the expected performance score."""

    if score >= 85.0:
        return "Excellent"
    if score >= 70.0:
        return "Strong"
    if score >= 55.0:
        return "Good"
    return "Limited"


def fee_text(row: pd.Series) -> str:
    """Format the transaction fee with the source local-currency code when present."""

    tier = str(row.get("amount_tier", "cc1"))
    currency_column = f"{tier} lcu code"
    currency = str(row.get(currency_column, "LCU") or "LCU")
    return f"{float(row['fee_lcu']):,.2f} {currency}"


def render_consumer_recommendation(
    recommendation: pd.Series,
    model_candidates: pd.DataFrame,
    inference: object,
    amount_match: object,
) -> None:
    """Render the consumer-facing optimized remittance recommendation."""

    performance = performance_score(recommendation)
    st.subheader("Recommended Transfer Service")
    col1, col2, col3 = st.columns(3)
    col1.metric("Service provider", str(recommendation["firm"]))
    col2.metric("Transfer method", str(recommendation["payment instrument"]))
    col3.metric("Receiving method", str(recommendation["pickup method"]))

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Transaction fee", fee_text(recommendation))
    col5.metric("FX spread", f"{float(recommendation['fx_margin']):.2f}%")
    col6.metric("Settlement time", str(recommendation["speed actual"]))
    col7.metric(
        "Overall performance",
        f"{performance:.1f}%",
        help=performance_band(performance),
    )

    st.caption(
        f"Matched transfer amount {float(amount_match.requested_amount):,.2f} to "
        f"{amount_match.amount_tier.upper()} historical denomination "
        f"{float(amount_match.matched_denomination):,.2f}. Internal use-case profile: "
        f"{inference.profile_label}."
    )

    if bool(recommendation.get("batch_mode", False)):
        st.subheader("Batch Assignment")
        st.caption(
            f"{int(recommendation.get('batch_transfer_count', 1))} transfers optimized together. "
            f"Provider counts: {recommendation.get('batch_provider_counts', {})}"
        )
        st.dataframe(
            batch_assignment_frame(recommendation),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Available Feasible Solutions")
    st.caption("Every row is a valid solution in the verified optimization model.")
    feasible = feasible_solution_frame(model_candidates)
    st.dataframe(
        feasible.style.format(
            {
                "risk_loss": "{:.3f}",
                "overall_performance": "{:.1f}%",
                "weighted_score": "{:.6f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def batch_assignment_frame(recommendation: pd.Series) -> pd.DataFrame:
    """Split a selected batch plan into readable assignment rows."""

    raw_services = str(recommendation.get("batch_selected_services", ""))
    services = [service for service in raw_services.split(" || ") if service]
    return pd.DataFrame(
        [
            {"transfer": number, "selected_service": service}
            for number, service in enumerate(services, start=1)
        ]
    )


def render_corridor_observatory(
    raw: pd.DataFrame,
    source: str,
    destination: str,
    transfer_amount: float,
) -> None:
    """Render an interactive corridor panel instead of a raw CSV preview."""

    try:
        amount_match = match_transfer_amount(raw, float(transfer_amount))
    except ValueError as exc:
        st.error(str(exc))
        return

    filtered = filter_dataset(
        raw,
        FilterSpec(source_name=source, destination_name=destination),
    )
    prepared = prepare_candidates(filtered, amount_tier=amount_match.amount_tier)
    service_options = latest_service_options(prepared)
    loss_candidates = add_objective_losses(service_options)
    pareto_options = pareto_prune(loss_candidates)

    st.subheader("Corridor Observatory")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Historical records", f"{len(filtered):,}")
    metric_cols[1].metric("Service options", f"{len(service_options):,}")
    metric_cols[2].metric("Pareto services", f"{len(pareto_options):,}")
    metric_cols[3].metric("Matched tier", amount_match.amount_tier.upper())

    st.caption(
        f"{source} to {destination} | requested amount {float(transfer_amount):,.2f} | "
        f"historical denomination {float(amount_match.matched_denomination):,.2f}"
    )

    if service_options.empty:
        st.warning("No services are available for this corridor and transfer amount.")
        return

    chart_tab, channel_tab, service_tab = st.tabs(
        ["Cost Landscape", "Service Mix", "Service Inspector"]
    )

    with chart_tab:
        cost_cols = st.columns(3)
        cost_cols[0].metric("Lowest fee", fee_text(service_options.sort_values("fee_lcu").iloc[0]))
        cost_cols[1].metric("Lowest FX spread", f"{float(service_options['fx_margin'].min()):.2f}%")
        cost_cols[2].metric("Fastest score", f"{float(service_options['speed_score'].max()):.2f}")

        cost_chart = service_options.copy()
        cost_chart["service"] = (
            cost_chart["firm"].astype(str)
            + " | "
            + cost_chart["payment instrument"].astype(str)
        )
        cost_chart = (
            cost_chart.sort_values(["fee_lcu", "fx_margin"], kind="mergesort")
            .head(12)
            .loc[:, ["service", "fee_lcu", "fx_margin"]]
            .set_index("service")
        )
        st.bar_chart(cost_chart)

    with channel_tab:
        mix_left, mix_right = st.columns(2)
        with mix_left:
            payment_mix = (
                service_options["payment instrument"]
                .value_counts()
                .rename_axis("payment_method")
                .reset_index(name="services")
                .set_index("payment_method")
            )
            st.bar_chart(payment_mix)
        with mix_right:
            receiving_mix = (
                service_options["pickup method"]
                .value_counts()
                .rename_axis("receiving_method")
                .reset_index(name="services")
                .set_index("receiving_method")
            )
            st.bar_chart(receiving_mix)

        speed_mix = (
            service_options["speed actual"]
            .value_counts()
            .rename_axis("settlement_time")
            .reset_index(name="services")
            .set_index("settlement_time")
        )
        st.bar_chart(speed_mix)

    with service_tab:
        indexed = service_options.reset_index(drop=True).copy()
        indexed["display_name"] = (
            indexed["firm"].astype(str)
            + " | "
            + indexed["payment instrument"].astype(str)
            + " | "
            + indexed["pickup method"].astype(str)
        )
        selected_name = st.selectbox("Service option", indexed["display_name"].tolist())
        selected = indexed.loc[indexed["display_name"].eq(selected_name)].iloc[0]

        selected_cols = st.columns(4)
        selected_cols[0].metric("Provider", str(selected["firm"]))
        selected_cols[1].metric("Fee", fee_text(selected))
        selected_cols[2].metric("FX spread", f"{float(selected['fx_margin']):.2f}%")
        selected_cols[3].metric("Settlement", str(selected["speed actual"]))
        st.caption(
            f"Payment method: {selected['payment instrument']} | "
            f"Receiving method: {selected['pickup method']} | Period: {selected['period']}"
        )


def render_metric_charts(metrics: pd.DataFrame) -> None:
    """Render graphical benchmark comparisons for the research dashboard."""

    chart_columns = [
        "feasibility_rate",
        "objective_value",
        "relative_optimality_gap",
        "optimum_hit_probability",
        "end_to_end_runtime_s",
        "stability",
        "time_to_solution_s",
    ]
    chart_tabs = st.tabs(
        [
            "Feasibility",
            "Objective",
            "Gap",
            "Optimum Hit",
            "Runtime",
            "Stability",
            "Time To Solution",
        ]
    )
    for tab, column in zip(chart_tabs, chart_columns):
        with tab:
            chart_data = (
                metrics[["algorithm", column]]
                .replace([np.inf, -np.inf], np.nan)
                .set_index("algorithm")
            )
            st.bar_chart(chart_data)


def render_research_dashboard(
    metrics: pd.DataFrame,
    solver_validations: list[dict[str, object]],
    execution_failures: list[dict[str, object]],
    results: list[dict[str, object]],
    model_candidates: pd.DataFrame,
    profiled_candidates: pd.DataFrame,
    inference: object,
    amount_match: object,
    verification: dict[str, object],
    qubo: object,
    policy_report: dict[str, object],
    batch_report: dict[str, object],
) -> None:
    """Render solver metrics, ML profile details, QUBO validation, and circuit output."""

    render_metric_cards(metrics)
    st.subheader("Metric Graphs")
    render_metric_charts(metrics)

    st.subheader("Inferred Use-Case Policy")
    profile_cols = st.columns(4)
    profile_cols[0].metric("Profile", inference.profile_label)
    profile_cols[1].metric("Confidence", f"{float(inference.confidence):.1%}")
    profile_cols[2].metric("K-Means clusters", int(inference.cluster_count))
    profile_cols[3].metric("Amount tier", amount_match.amount_tier.upper())
    st.dataframe(profile_weight_table(inference), width="stretch", hide_index=True)

    probability_frame = pd.DataFrame(
        [
            {"profile": profile, "probability": probability}
            for profile, probability in inference.probabilities.items()
        ]
    )
    if not probability_frame.empty:
        st.bar_chart(probability_frame.set_index("profile"))

    summary = profile_summary(profiled_candidates)
    if not summary.empty:
        st.subheader("K-Means Service Profiles")
        st.dataframe(summary, width="stretch", hide_index=True)

    if execution_failures:
        st.warning("Some solvers did not execute successfully")
        st.dataframe(pd.DataFrame(execution_failures), width="stretch", hide_index=True)

    with st.expander("Policy and batch constraints"):
        report_col, batch_col = st.columns(2)
        with report_col:
            st.caption("Hard eligibility constraints")
            st.json(policy_report)
        with batch_col:
            st.caption("Batch/provider concentration constraints")
            st.json(batch_report)

    st.subheader("Core Metrics")
    st.dataframe(
        metrics.style.format(
            {
                "feasibility_rate": "{:.2%}",
                "objective_value": "{:.6f}",
                "relative_optimality_gap": "{:.2%}",
                "optimum_hit_probability": "{:.2%}",
                "end_to_end_runtime_s": "{:.4f}",
                "stability": "{:.2%}",
                "time_to_solution_s": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Solver output validation"):
        st.dataframe(pd.DataFrame(solver_validations), width="stretch", hide_index=True)

    st.subheader("Selected solutions by solver")
    st.dataframe(solution_frame(results, model_candidates), width="stretch", hide_index=True)

    render_quantum_circuit(results)

    st.subheader("QUBO candidate set")
    columns = [
        "candidate_index",
        "binary_state",
        "model_source",
        "service_profile_label",
        "firm",
        "corridor",
        "period",
        "payment instrument",
        "pickup method",
        "speed actual",
        "total_cost_pct",
        "fee_lcu",
        "fx_margin",
        "transaction_fee_loss",
        "time_loss",
        "fx_spread_loss",
        "risk_loss",
        "coverage_loss",
        "transparency_loss",
        "access_point_loss",
        "settlement_days",
        "batch_transfer_count",
        "batch_provider_counts",
        "batch_selected_services",
        "weighted_score",
    ]
    visible_columns = [column for column in columns if column in model_candidates.columns]
    st.dataframe(model_candidates[visible_columns], width="stretch", hide_index=True)

    with st.expander("QUBO verification"):
        st.json(verification)
        if qubo.size <= 16:
            st.dataframe(
                pd.DataFrame(qubo.matrix, columns=qubo.labels, index=qubo.labels),
                width="stretch",
            )


def render_metric_cards(metrics: pd.DataFrame) -> None:
    """Render top-level benchmark metrics."""

    if metrics.empty:
        return
    best_row = rank_metrics(metrics).iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best ranked algorithm", str(best_row["algorithm"]))
    col2.metric("Best objective", f"{best_row['objective_value']:.4f}")
    col3.metric("Best gap", f"{best_row['relative_optimality_gap']:.2%}")
    col4.metric("Sample runs", f"{int(metrics['runs'].max()):,}")


def selected_quantum_backend(label: str) -> dict[str, str]:
    """Return backend settings for the selected simulator option."""

    return QUANTUM_BACKEND_OPTIONS[label]


def render_quantum_circuit(results: list[dict[str, object]]) -> None:
    """Show the final QAOA circuit produced by the quantum optimizer."""

    quantum_result = next(
        (result for result in results if result.get("algorithm") == "QAOA"),
        None,
    )
    if quantum_result is None:
        return

    st.subheader("Quantum Circuit")
    if quantum_result.get("status") != "ok":
        st.info(str(quantum_result.get("note", "QAOA did not run.")))
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Qubits", int(quantum_result.get("circuit_num_qubits", 0)))
    col2.metric("Candidate rows", int(quantum_result.get("candidate_count", 0)))
    col3.metric("Basis states", int(quantum_result.get("basis_state_count", 0)))
    col4.metric("Circuit iterations", int(quantum_result.get("execution_iterations", 1)))
    col5.metric("Depth", int(quantum_result.get("circuit_depth", 0)))
    st.caption(
        f"Optimized locally with {quantum_result.get('optimization_backend', 'unknown')} | "
        f"Final execution: {quantum_result.get('execution_backend', 'unknown')} | "
        f"Total shots: {int(quantum_result.get('total_shots', 0))} | "
        f"Job: {quantum_result.get('job_id') or 'local/no job id'}"
    )

    st.code(str(quantum_result.get("circuit_diagram", "")), language="text")
    with st.expander("Circuit parameters and gates"):
        st.json(
            {
                "optimized_parameters": quantum_result.get("parameters", []),
                "gate_counts": quantum_result.get("circuit_gate_counts", {}),
                "measurement_counts": quantum_result.get("measurement_counts", {}),
                "execution_iterations": quantum_result.get("execution_iterations", 1),
                "shots_per_iteration": quantum_result.get("shots_per_iteration", 0),
                "total_shots": quantum_result.get("total_shots", 0),
                "job_ids": quantum_result.get("job_ids", []),
                "note": quantum_result.get("note", ""),
            }
        )


def main() -> None:
    """Run the Streamlit UI."""

    render_app_header()

    data_path = Path(DEFAULT_DATA_PATH)
    data_stat = data_path.stat()
    raw = cached_dataset(str(data_path), data_stat.st_mtime_ns, data_stat.st_size)

    with st.sidebar:
        st.header("Transfer")
        source = required_selectbox("Source country", unique_values(raw, "source_name"))
        source_scope = raw[raw["source_name"] == source]
        destination = required_selectbox("Destination country", unique_values(source_scope, "destination_name"))
        transfer_amount = st.number_input(
            "Transfer amount",
            min_value=1.0,
            value=200.0,
            step=50.0,
        )

        with st.expander("Research settings"):
            st.caption(
                f"QUBO candidate cap: {MODEL_CANDIDATE_CAP}. "
                "QAOA qubits use compact binary indexing: ceil(log2(candidate rows))."
            )
            st.markdown("Policy constraints")
            max_total_cost_pct = st.number_input(
                "Max total cost %",
                min_value=0.0,
                value=100.0,
                step=0.5,
            )
            max_settlement_days = st.number_input(
                "Max settlement days",
                min_value=0.0,
                value=5.0,
                step=0.5,
            )
            require_transparency = st.checkbox("Require transparent pricing", value=False)
            min_coverage_score = st.slider("Minimum network coverage score", 0.0, 1.0, 0.0, 0.05)
            access_point_options = unique_values(raw, "access point")
            selected_access_points = st.multiselect(
                "Allowed access points",
                access_point_options,
                default=access_point_options,
            )

            st.markdown("Batch research")
            batch_transfer_count = st.number_input(
                "Batch transfers",
                min_value=1,
                max_value=4,
                value=1,
                step=1,
                help="Optimize several transfers together for provider-concentration tests.",
            )
            max_provider_share_pct = st.slider(
                "Max provider concentration %",
                min_value=25,
                max_value=100,
                value=100,
                step=5,
            )
            batch_candidate_cap = st.slider(
                "Batch candidate cap",
                2,
                MODEL_CANDIDATE_CAP,
                min(DEFAULT_BATCH_CANDIDATE_CAP, MODEL_CANDIDATE_CAP),
                1,
            )
            batch_plan_cap = st.slider("Batch plan cap", 8, 256, DEFAULT_BATCH_PLAN_CAP, 8)

            st.markdown("Solver controls")
            penalty_multiplier = st.slider("QUBO penalty multiplier", 1.1, 5.0, 2.0, 0.1)
            sa_reads = st.slider("Annealing reads", 16, 512, 128, 16)
            sa_sweeps = st.slider("Annealing sweeps", 100, 3000, 600, 100)
            run_quantum = st.checkbox("Run QAOA", value=True)
            quantum_backend_label = st.selectbox(
                "Quantum execution backend",
                list(QUANTUM_BACKEND_OPTIONS),
            )
            quantum_backend = selected_quantum_backend(quantum_backend_label)
            selected_device_id = st.text_input(
                "Device id",
                value=quantum_backend["device_id"],
                disabled=True,
                key=f"device_id_{quantum_backend['backend_name']}",
            )
            qaoa_reps = st.slider("QAOA depth", 1, 3, 1)
            qaoa_shots = st.slider("QAOA shots", 64, 4096, 512, 64)
            qaoa_circuit_iterations = st.number_input(
                "QAOA circuit iterations",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
                help="How many times to execute the final optimized QAOA circuit.",
            )
            qaoa_optimizer_iterations = st.slider("QAOA optimizer iterations", 8, 120, 40, 4)
            qbraid_timeout_s = st.number_input("qBraid timeout seconds", value=300, min_value=30)
            seed = st.number_input("Random seed", value=42, min_value=0, step=1)

        run_button = st.button("Run optimization", type="primary", width="stretch")

    qbraid = qbraid_status()
    st.caption(
        f"qBraid enabled: {qbraid.get('enabled', False)} | "
        f"API key present: {qbraid.get('api_key_present', False)} | "
        f"SDK: {qbraid.get('sdk_available', False)} ({qbraid.get('sdk_version', 'unknown')}) | "
        f".env loaded: {qbraid.get('env_file_loaded', False)} | "
        f"target: {qbraid.get('provider', 'local-qiskit')} / "
        f"{qbraid.get('device_id', QBRAID_SIMULATOR_DEVICE_ID)}"
    )
    active_access_points = (
        tuple(selected_access_points)
        if access_point_options and len(selected_access_points) < len(access_point_options)
        else ()
    )
    policy_constraints = PolicyConstraints(
        max_total_cost_pct=float(max_total_cost_pct),
        max_settlement_days=float(max_settlement_days),
        require_transparency=bool(require_transparency),
        min_coverage_score=float(min_coverage_score),
        allowed_access_points=active_access_points,
        max_provider_share=float(max_provider_share_pct) / 100.0,
    )
    batch_settings = BatchSettings(
        transfer_count=int(batch_transfer_count),
        max_provider_share=float(max_provider_share_pct) / 100.0,
        candidate_cap=int(batch_candidate_cap),
        plan_cap=int(batch_plan_cap),
    )

    if not run_button:
        render_corridor_observatory(raw, source, destination, float(transfer_amount))
        return

    run_status = st.status("Running optimization", state="running", expanded=True)
    run_status.write("Matching the transfer amount to the closest historical tier.")
    try:
        amount_match = match_transfer_amount(raw, float(transfer_amount))
    except ValueError as exc:
        run_status.update(label="Optimization stopped", state="error", expanded=True)
        st.error(str(exc))
        return

    run_status.write("Filtering the corridor and preparing service candidates.")
    spec = FilterSpec(
        source_name=source,
        destination_name=destination,
    )

    filtered = filter_dataset(raw, spec)
    prepared = prepare_candidates(filtered, amount_tier=amount_match.amount_tier)
    service_options = latest_service_options(prepared)
    loss_candidates = add_objective_losses(service_options)
    constrained_candidates, policy_report = apply_policy_constraints(
        loss_candidates,
        policy_constraints,
    )
    pruned = pareto_prune(constrained_candidates)
    run_status.write("Inferring the internal use-case policy from service profiles.")
    profiled_candidates, inference = infer_use_case_policy(
        pruned,
        transfer_amount=float(transfer_amount),
        random_state=int(seed),
    )
    scored = score_candidates(profiled_candidates, inference.weights)
    service_model_candidates, _ = select_qubo_candidates(scored, MODEL_CANDIDATE_CAP)
    model_candidates, batch_report = build_batch_plans(service_model_candidates, batch_settings)
    model_candidates["candidate_index"] = np.arange(len(model_candidates), dtype=int)
    qubit_count = required_qubits(len(model_candidates)) if not model_candidates.empty else 0
    basis_state_count = 2**qubit_count if qubit_count else 0
    if qubit_count:
        model_candidates["binary_state"] = [
            format(int(index), f"0{qubit_count}b")
            for index in model_candidates["candidate_index"]
        ]

    count_cols = st.columns(6)
    count_cols[0].metric("Filtered rows", f"{len(filtered):,}")
    count_cols[1].metric("Service options", f"{len(service_options):,}")
    count_cols[2].metric("Eligible services", f"{len(constrained_candidates):,}")
    count_cols[3].metric("Pareto rows", f"{len(pruned):,}")
    count_cols[4].metric("Model candidates", f"{len(model_candidates):,}")
    count_cols[5].metric("QAOA qubits", f"{qubit_count:,}")

    if model_candidates.empty:
        run_status.update(label="Optimization stopped", state="error", expanded=True)
        st.error(
            "No candidates remain after corridor, amount, policy constraints, "
            "Pareto filtering, and batch planning."
        )
        with st.expander("Constraint details", expanded=True):
            st.json({"policy": policy_report, "batch": batch_report})
        return

    st.caption(
        f"Compact QUBO encoding: {len(model_candidates):,} candidates use "
        f"{qubit_count:,} qubits, giving {basis_state_count:,} possible basis states."
    )
    run_status.write("Building the QUBO and validating it against the constrained model.")
    scores = model_candidates["weighted_score"].to_numpy(dtype=float)
    labels = [
        str(row.get("batch_selected_services"))
        if row.get("batch_selected_services")
        else f"{row['firm']} | {row['payment instrument']} | {row['pickup method']}"
        for _, row in model_candidates.iterrows()
    ]
    base_penalty = max(1.0, float(scores.max()))
    qubo = build_selection_qubo(
        scores,
        penalty=penalty_multiplier * base_penalty,
        labels=labels,
    )
    verification = verify_qubo_equivalence(qubo)

    if verification["equivalent"]:
        st.success(verification["message"])
    else:
        st.error("QUBO Validation failed")
        with st.expander("Verification details", expanded=True):
            st.json(verification)
        run_status.update(label="QUBO validation failed", state="error", expanded=True)
        return

    run_status.write("Running classical baselines and heuristics.")
    exact = solve_original_exact(scores)
    results: list[dict[str, object]] = [
        business_as_usual_baseline(model_candidates),
        exact_mathematical_baseline(model_candidates),
        lowest_fee_heuristic(model_candidates),
        fastest_transfer_heuristic(model_candidates),
        lowest_fx_heuristic(model_candidates),
        simulated_annealing(
            qubo,
            num_reads=sa_reads,
            sweeps=sa_sweeps,
            seed=int(seed),
        ),
    ]

    if run_quantum:
        run_status.write(
            f"Running the QAOA quantum circuit for {int(qaoa_circuit_iterations)} iteration(s)."
        )
        with st.spinner("Running QAOA circuit"):
            results.append(
                run_qaoa(
                    qubo,
                    reps=qaoa_reps,
                    shots=qaoa_shots,
                    max_qubits=MODEL_CANDIDATE_CAP,
                    seed=int(seed),
                    optimizer_maxiter=int(qaoa_optimizer_iterations),
                    execution_iterations=int(qaoa_circuit_iterations),
                    backend_name=quantum_backend["backend_name"],
                    qbraid_device_id=selected_device_id,
                    qbraid_timeout_s=int(qbraid_timeout_s),
                )
            )

    run_status.write("Validating solver outputs and computing benchmark metrics.")
    execution_failures = [
        {
            "algorithm": result.get("algorithm", "unknown"),
            "status": result.get("status", "failed"),
            "note": result.get("note", ""),
        }
        for result in results
        if result.get("status", "ok") != "ok"
    ]
    solver_outputs = [
        result
        for result in results
        if result.get("status", "ok") == "ok" and result.get("samples")
    ]
    solver_validations = validate_solver_outputs(solver_outputs, scores)
    invalid_outputs = [
        validation for validation in solver_validations if not validation["valid"]
    ]
    if invalid_outputs:
        run_status.update(label="Solver validation failed", state="error", expanded=True)
        st.error("Solver output validation failed")
        st.dataframe(pd.DataFrame(solver_validations), width="stretch", hide_index=True)
        render_quantum_circuit(results)
        return

    metrics = metrics_frame(
        [
            summarize_samples(
                result["algorithm"],
                result.get("samples", []),
                float(exact["objective"]),
                exact["indices"],
                float(result.get("runtime_s", 0.0)),
            )
            for result in results
        ]
    )
    run_status.update(label="Optimization complete", state="complete", expanded=False)

    recommendation = model_candidates.iloc[int(exact["index"])]
    consumer_tab, research_tab = st.tabs(["Consumer recommendation", "Research dashboard"])
    with consumer_tab:
        render_consumer_recommendation(
            recommendation,
            model_candidates,
            inference,
            amount_match,
        )
    with research_tab:
        render_research_dashboard(
            metrics=metrics,
            solver_validations=solver_validations,
            execution_failures=execution_failures,
            results=results,
            model_candidates=model_candidates,
            profiled_candidates=scored,
            inference=inference,
            amount_match=amount_match,
            verification=verification,
            qubo=qubo,
            policy_report=policy_report,
            batch_report=batch_report,
        )


if __name__ == "__main__":
    main()
