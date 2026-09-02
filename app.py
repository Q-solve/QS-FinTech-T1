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
from qkash.classical import (
    business_as_usual_baseline,
    exact_mathematical_baseline,
    simulated_annealing,
)
from qkash.data import (
    DEFAULT_DATA_PATH,
    FilterSpec,
    filter_dataset,
    load_dataset,
    prepare_candidates,
    unique_values,
)
from qkash.qbraid_bridge import qbraid_status
from qkash.quantum import run_qaoa
from qkash.scoring import (
    build_selection_qubo,
    normalize_weights,
    parse_weight_query,
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

# DEFAULT_PRIORITY_QUERY seeds the three supported objective weights.
DEFAULT_PRIORITY_QUERY = "low transaction fee, fast transfer time, low FX spread"

# TARGET_QUBITS fixes the candidate count and QAOA qubit count.
TARGET_QUBITS = 5

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


def inject_app_styles() -> None:
    """Install small UI styles, including the custom optimization indicator."""

    st.markdown(
        """
        <style>
        div[data-testid="stStatusWidget"] {
            display: none !important;
        }

        .qkash-run-indicator {
            align-items: center;
            background: #f8fafc;
            border: 1px solid #d7dee8;
            border-radius: 8px;
            display: flex;
            gap: 18px;
            margin: 8px 0 18px;
            max-width: 430px;
            padding: 14px 16px;
        }

        .qkash-scene {
            height: 76px;
            position: relative;
            width: 162px;
        }

        .qkash-cat-body {
            background: #1f2933;
            border-radius: 50% 48% 42% 44%;
            bottom: 14px;
            height: 30px;
            left: 10px;
            position: absolute;
            width: 58px;
        }

        .qkash-cat-head {
            background: #1f2933;
            border-radius: 50%;
            bottom: 33px;
            height: 30px;
            left: 53px;
            position: absolute;
            width: 30px;
        }

        .qkash-cat-head::before,
        .qkash-cat-head::after {
            border-bottom: 12px solid #1f2933;
            border-left: 7px solid transparent;
            border-right: 7px solid transparent;
            content: "";
            position: absolute;
            top: -7px;
        }

        .qkash-cat-head::before {
            left: 1px;
            transform: rotate(-18deg);
        }

        .qkash-cat-head::after {
            right: 1px;
            transform: rotate(18deg);
        }

        .qkash-cat-tail {
            border: 6px solid #1f2933;
            border-left: 0;
            border-radius: 0 22px 22px 0;
            bottom: 26px;
            height: 32px;
            left: 2px;
            position: absolute;
            transform-origin: 53px 28px;
            width: 35px;
            animation: qkash-tail 1.2s ease-in-out infinite;
        }

        .qkash-cat-leg {
            background: #1f2933;
            border-radius: 0 0 6px 6px;
            bottom: 4px;
            height: 14px;
            position: absolute;
            width: 7px;
        }

        .qkash-cat-leg.front {
            left: 56px;
        }

        .qkash-cat-leg.back {
            left: 24px;
        }

        .qkash-box {
            background: #b9783f;
            border: 2px solid #67401f;
            bottom: 8px;
            height: 40px;
            left: 96px;
            position: absolute;
            width: 54px;
            animation: qkash-box 1.8s ease-in-out infinite;
        }

        .qkash-box::before {
            background: #d09354;
            border: 2px solid #67401f;
            content: "";
            height: 12px;
            left: -5px;
            position: absolute;
            top: -16px;
            width: 60px;
        }

        .qkash-box::after {
            background: rgba(255, 255, 255, 0.24);
            content: "";
            height: 40px;
            left: 25px;
            position: absolute;
            top: 0;
            width: 2px;
        }

        .qkash-run-text {
            color: #111827;
            font-size: 0.94rem;
            font-weight: 600;
            line-height: 1.35;
        }

        .qkash-run-text span {
            color: #536171;
            display: block;
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 2px;
        }

        @keyframes qkash-tail {
            0%, 100% { transform: rotate(-5deg); }
            50% { transform: rotate(9deg); }
        }

        @keyframes qkash-box {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-2px); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def render_run_indicator() -> st.delta_generator.DeltaGenerator:
    """Show the custom run-state visual while optimization is executing."""

    placeholder = st.empty()
    placeholder.markdown(
        """
        <div class="qkash-run-indicator" role="status" aria-live="polite">
            <div class="qkash-scene" aria-hidden="true">
                <div class="qkash-cat-tail"></div>
                <div class="qkash-cat-body"></div>
                <div class="qkash-cat-head"></div>
                <div class="qkash-cat-leg back"></div>
                <div class="qkash-cat-leg front"></div>
                <div class="qkash-box"></div>
            </div>
            <div class="qkash-run-text">
                Optimization running
                <span>Preparing the QUBO and quantum circuit.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return placeholder


def required_selectbox(label: str, options: list[str]) -> str:
    """Render a required selectbox and stop if no valid option exists."""

    if not options:
        st.error(f"No available options for {label}.")
        st.stop()
    return st.selectbox(label, options)


def token_match_mask(series: pd.Series, selected: str) -> pd.Series:
    """Match one selected UI value against comma-separated CSV fields."""

    wanted = selected.strip().lower()

    def matches(raw_value: object) -> bool:
        tokens = {token.strip().lower() for token in str(raw_value).split(",")}
        return wanted in tokens

    return series.fillna("").astype(str).map(matches)


def display_candidate(row: pd.Series) -> dict[str, object]:
    """Select the candidate fields shown in solution tables."""

    return {
        "firm": row["firm"],
        "corridor": row["corridor"],
        "period": row["period"],
        "firm_type": row["firm_type"],
        "pickup": row["pickup method"],
        "speed": row["speed actual"],
        "coverage": row["receiving network coverage"],
        "transparent": row["transparent"],
        "total_cost_%": row["total_cost_pct"],
        "fee_lcu": row["fee_lcu"],
        "fx_margin": row["fx_margin"],
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


def weight_label(name: str) -> str:
    """Return human-readable labels for the three weight sliders."""

    labels = {
        "transaction_fee": "Transaction fee",
        "time": "Time",
        "fx_spread": "FX spread",
    }
    return labels.get(name, name.replace("_", " ").title())


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

    st.subheader("Quantum circuit used")
    if quantum_result.get("status") != "ok":
        st.info(str(quantum_result.get("note", "QAOA did not run.")))
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Qubits", int(quantum_result.get("circuit_num_qubits", 0)))
    col2.metric("Depth", int(quantum_result.get("circuit_depth", 0)))
    col3.metric("Width", int(quantum_result.get("circuit_width", 0)))
    st.caption(
        f"Optimized locally with {quantum_result.get('optimization_backend', 'unknown')} | "
        f"Final execution: {quantum_result.get('execution_backend', 'unknown')} | "
        f"Job: {quantum_result.get('job_id') or 'local/no job id'}"
    )

    st.code(str(quantum_result.get("circuit_diagram", "")), language="text")
    with st.expander("Circuit parameters and gates"):
        st.json(
            {
                "optimized_parameters": quantum_result.get("parameters", []),
                "gate_counts": quantum_result.get("circuit_gate_counts", {}),
                "measurement_counts": quantum_result.get("measurement_counts", {}),
                "note": quantum_result.get("note", ""),
            }
        )


def main() -> None:
    """Run the Streamlit UI."""

    inject_app_styles()
    render_app_header()

    data_path = Path(DEFAULT_DATA_PATH)
    data_stat = data_path.stat()
    raw = cached_dataset(str(data_path), data_stat.st_mtime_ns, data_stat.st_size)

    with st.sidebar:
        st.header("Filters")
        source = required_selectbox("Source", unique_values(raw, "source_name"))
        source_scope = raw[raw["source_name"] == source]

        destination = required_selectbox(
            "Destination",
            unique_values(source_scope, "destination_name"),
        )
        destination_scope = source_scope[source_scope["destination_name"] == destination]

        payment_instrument = required_selectbox(
            "Payment instrument",
            unique_values(destination_scope, "payment instrument"),
        )
        payment_scope = destination_scope[
            token_match_mask(
                destination_scope["payment instrument"],
                payment_instrument,
            )
        ]

        pickup_method = required_selectbox(
            "Pickup method",
            unique_values(payment_scope, "pickup method"),
        )

        st.header("Objective")
        query = st.text_area(
            "Priority query",
            value=DEFAULT_PRIORITY_QUERY,
            height=80,
        )
        query_weights = parse_weight_query(query)

        weight_inputs: dict[str, float] = {}
        for name, value in query_weights.items():
            weight_inputs[name] = st.slider(
                f"{weight_label(name)} weight",
                min_value=0.0,
                max_value=1.0,
                value=float(value),
                step=0.01,
            )
        weights = normalize_weights(weight_inputs)

        amount_tier = st.radio(
            "Amount tier",
            ["cc1", "cc2"],
            format_func=lambda value: "cc1 - 200 denomination"
            if value == "cc1"
            else "cc2 - 500 denomination",
            horizontal=True,
        )
        latest_per_firm = st.checkbox("Latest row per firm", value=False)

        st.header("Optimization")
        st.caption(f"QAOA system size: {TARGET_QUBITS} candidates / {TARGET_QUBITS} qubits")
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

    if not run_button:
        preview = raw.head(20)
        st.dataframe(preview, width="stretch", hide_index=True)
        return

    run_indicator = render_run_indicator()
    spec = FilterSpec(
        source_name=source,
        destination_name=destination,
        pickup_method=pickup_method,
        payment_instrument=payment_instrument,
    )

    filtered = filter_dataset(raw, spec)
    prepared = prepare_candidates(filtered, amount_tier=amount_tier, latest_per_firm=latest_per_firm)
    scored = score_candidates(prepared, weights)
    model_candidates, pruned = select_qubo_candidates(scored, TARGET_QUBITS)
    model_candidates["candidate_index"] = np.arange(len(model_candidates), dtype=int)

    count_cols = st.columns(4)
    count_cols[0].metric("Filtered rows", f"{len(filtered):,}")
    count_cols[1].metric("Scored rows", f"{len(scored):,}")
    count_cols[2].metric("Pareto rows", f"{len(pruned):,}")
    count_cols[3].metric("Model variables", f"{len(model_candidates):,} / {TARGET_QUBITS}")

    if model_candidates.empty:
        run_indicator.empty()
        st.error("No candidates remain after filtering and scoring.")
        return
    if len(model_candidates) < TARGET_QUBITS:
        run_indicator.empty()
        st.error(
            f"A valid {TARGET_QUBITS}-qubit QAOA run needs {TARGET_QUBITS} "
            "real candidates after filtering and "
            f"scoring. This selection has {len(model_candidates)}. Broaden the filters "
            f"or turn off latest-row-per-firm to produce a {TARGET_QUBITS}-variable QUBO."
        )
        return
    fallback_count = int(
        model_candidates["model_source"].eq("Best scored fallback").sum()
        if "model_source" in model_candidates.columns
        else 0
    )
    if fallback_count:
        st.info(
            f"Strict Pareto pruning kept {len(pruned):,} candidate(s). Added "
            f"{fallback_count} next-best scored candidate(s) to form the fixed "
            f"{TARGET_QUBITS}-qubit QUBO."
        )

    scores = model_candidates["weighted_score"].to_numpy(dtype=float)
    labels = [
        f"{row.firm} | {row.corridor} | {row.period}"
        for row in model_candidates.itertuples(index=False)
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
        run_indicator.empty()
        return

    exact = solve_original_exact(scores)
    results: list[dict[str, object]] = [
        business_as_usual_baseline(model_candidates),
        exact_mathematical_baseline(model_candidates),
        simulated_annealing(
            qubo,
            num_reads=sa_reads,
            sweeps=sa_sweeps,
            seed=int(seed),
        ),
    ]

    if run_quantum:
        results.append(
            run_qaoa(
                qubo,
                reps=qaoa_reps,
                shots=qaoa_shots,
                max_qubits=TARGET_QUBITS,
                seed=int(seed),
                backend_name=quantum_backend["backend_name"],
                qbraid_device_id=selected_device_id,
                qbraid_timeout_s=int(qbraid_timeout_s),
            )
        )

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
        run_indicator.empty()
        st.error("Solver output validation failed")
        st.dataframe(pd.DataFrame(solver_validations), width="stretch", hide_index=True)
        render_quantum_circuit(results)
        return

    run_indicator.empty()
    if execution_failures:
        st.warning("Some solvers did not execute successfully")
        st.dataframe(pd.DataFrame(execution_failures), width="stretch", hide_index=True)

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

    render_metric_cards(metrics)

    with st.expander("Solver output validation"):
        st.dataframe(pd.DataFrame(solver_validations), width="stretch", hide_index=True)

    st.subheader("Core metrics")
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

    st.subheader("Selected solutions")
    st.dataframe(
        solution_frame(results, model_candidates),
        width="stretch",
        hide_index=True,
    )

    render_quantum_circuit(results)

    st.subheader("QUBO candidate set")
    columns = [
        "candidate_index",
        "model_source",
        "firm",
        "corridor",
        "period",
        "firm_type",
        "pickup method",
        "speed actual",
        "receiving network coverage",
        "transparent",
        "total_cost_pct",
        "fee_lcu",
        "fx_margin",
        "transaction_fee_loss",
        "time_loss",
        "fx_spread_loss",
        "weighted_score",
    ]
    st.dataframe(model_candidates[columns], width="stretch", hide_index=True)

    with st.expander("QUBO verification"):
        st.json(verification)
        if qubo.size <= 16:
            st.dataframe(
                pd.DataFrame(qubo.matrix, columns=qubo.labels, index=qubo.labels),
                width="stretch",
            )


if __name__ == "__main__":
    main()
