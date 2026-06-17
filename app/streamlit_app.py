"""
Heart Disease Bayesian Network — Interactive PGM Demo
Deploy: streamlit run app/streamlit_app.py  |  Cloud: https://heartdiseasepredictiondemo.streamlit.app/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.presets import OPTIMIZED_PRESETS, PRESETS
from src.config import (
    FEATURE_LABELS,
    LEGACY_INPUT_GROUPS,
    OPTIMIZED_FEATURE_LABELS,
    OPTIMIZED_INPUT_GROUPS,
    OPTIMIZED_SEED,
    OPTIMIZED_STATE_DISPLAY,
    OPTIMIZED_STATE_LABELS,
    OPTIMIZED_THRESHOLD,
    OPTIMIZED_VARS,
    OUTPUT_DIR,
    STATE_LABELS,
    TARGET,
    ProjectConfig,
)
from src.dag_draw import draw_directed_dag
from src.data import (
    load_cached_or_build,
    load_cached_optimized,
    prepare_train_test,
    prepare_train_val_test,
    records_to_evidence,
)
from src.inference import (
    compare_inference,
    disease_probability,
    map_label,
    predict_disease,
    sensitivity_analysis,
)
from src.learning import (
    learn_manual_structure_bn,
    learn_naive_bayes_bn,
    learn_optimized_clinical_bn,
    learn_structure_and_parameters,
)

st.set_page_config(
    page_title="Heart Disease BN | PGM Medical Diagnosis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPORT_PATH = OUTPUT_DIR / "report.json"
APP_VERSION = "2.1-manual-structure-dag"

MODEL_CATALOG = [
    ("Optimized Clinical BN (recommended)", "optimized", True),
    ("Chow-Liu Tree BN", "tree", False),
    ("Naive Bayes BN", "naive", False),
    ("Manual Structure BN", "manual", False),
]


@st.cache_resource(show_spinner="Loading all 4 Bayesian Networks (first visit ~30s)…")
def load_all_models(_version: str = APP_VERSION) -> dict:
    """Train each BN once — every model is distinct (no shared placeholder)."""
    del _version  # bust cache when APP_VERSION changes
    opt_df = load_cached_optimized()
    opt_train, _, _ = prepare_train_val_test(opt_df, seed=OPTIMIZED_SEED)
    optimized = learn_optimized_clinical_bn(opt_train).model

    df = load_cached_or_build()
    train, _, _ = prepare_train_test(df)
    manual = learn_manual_structure_bn(train).model
    naive = learn_naive_bayes_bn(train).model
    tree = learn_structure_and_parameters(train, ProjectConfig(structure_learning_iters=40)).model

    return {"manual": manual, "naive": naive, "tree": tree, "optimized": optimized}


@st.cache_data
def load_report():
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return None


@st.cache_data
def load_test_sample(optimized: bool = False):
    if optimized:
        df = load_cached_optimized()
        _, _, test = prepare_train_val_test(df, seed=OPTIMIZED_SEED)
    else:
        df = load_cached_or_build()
        _, test, _ = prepare_train_test(df)
    return test


def _metric_pct(value) -> str:
    v = float(value)
    if v > 1.0:
        v /= 100.0
    return f"{v:.1%}"


def model_ui_config(network_name: str) -> tuple[dict, dict, dict, dict]:
    if "Optimized" in network_name:
        return (
            OPTIMIZED_STATE_LABELS,
            OPTIMIZED_FEATURE_LABELS,
            OPTIMIZED_PRESETS,
            OPTIMIZED_INPUT_GROUPS,
        )
    return STATE_LABELS, FEATURE_LABELS, PRESETS, LEGACY_INPUT_GROUPS


def _display_label(var: str, code: str, optimized: bool) -> str:
    if optimized and var in OPTIMIZED_STATE_DISPLAY:
        return OPTIMIZED_STATE_DISPLAY[var].get(code, code)
    return code


def _ordered_features(model, groups: dict[str, list[str]]) -> list[str]:
    ordered: list[str] = []
    for vars_in_group in groups.values():
        for v in vars_in_group:
            if v in model.nodes() and v != TARGET and v not in ordered:
                ordered.append(v)
    for n in model.nodes():
        if n != TARGET and n not in ordered:
            ordered.append(n)
    return ordered


def collect_evidence_for_model(
    model,
    network_name: str,
    key_prefix: str = "",
    preset_evidence: dict | None = None,
) -> dict[str, str]:
    """Grouped patient inputs with human-readable option labels."""
    state_labels, feature_labels, _, groups = model_ui_config(network_name)
    optimized = "Optimized" in network_name
    preset_evidence = preset_evidence or {}
    evidence: dict[str, str] = {}

    st.caption("Select clinical findings below. Each field maps to a node in the directed BN.")

    for group_name, _vars in groups.items():
        present = [v for v in _vars if v in model.nodes() and v != TARGET]
        if not present:
            continue
        with st.expander(f"**{group_name}**", expanded=True):
            cols = st.columns(2)
            for i, var in enumerate(present):
                model_states: list[str] = []
                for cpd in model.get_cpds():
                    if cpd.variable == var:
                        model_states = [str(s) for s in cpd.state_names[var]]
                        break
                options = model_states or state_labels.get(var, [])
                default_idx = 0
                if var in preset_evidence and preset_evidence[var] in options:
                    default_idx = options.index(preset_evidence[var])
                with cols[i % 2]:
                    evidence[var] = st.selectbox(
                        feature_labels.get(var, var),
                        options,
                        index=default_idx,
                        format_func=lambda x, v=var: _display_label(v, x, optimized),
                        key=f"{key_prefix}{network_name}_{var}",
                        help=f"BN node: `{var}`",
                    )
    return evidence


def evidence_summary_table(evidence: dict[str, str], network_name: str) -> pd.DataFrame:
    _, feature_labels, _, _ = model_ui_config(network_name)
    optimized = "Optimized" in network_name
    rows = []
    for var, code in evidence.items():
        rows.append({
            "Clinical variable": feature_labels.get(var, var),
            "Selected value": _display_label(var, code, optimized),
            "BN code": code,
        })
    return pd.DataFrame(rows)


def get_model_threshold(report: dict | None, network: str) -> float:
    if report and "metrics" in report:
        for row in report["metrics"]:
            name = row.get("Model", "")
            if "Optimized" in network and "Optimized" in name:
                return float(row.get("Threshold", OPTIMIZED_THRESHOLD))
            if "Chow-Liu" in network and "Chow-Liu" in name:
                return float(row.get("Threshold", 0.21))
            if "Naive" in network and "Naive" in name:
                return float(row.get("Threshold", 0.34))
            if "Manual" in network and "Manual" in name:
                return float(row.get("Threshold", 0.41))
    return OPTIMIZED_THRESHOLD if "Optimized" in network else 0.45


def classify_at_threshold(p_yes: float, threshold: float) -> str:
    return "Yes" if p_yes >= threshold else "No"


def risk_band(p: float) -> tuple[str, str]:
    if p < 0.25:
        return "Low risk", "#27ae60"
    if p < 0.55:
        return "Moderate risk", "#f39c12"
    return "High risk", "#e74c3c"


def primary_metrics(report: dict | None) -> list[dict]:
    if not report:
        return []
    if report.get("metrics_primary"):
        return report["metrics_primary"]
    return [m for m in report.get("metrics", []) if "Optimized" in m.get("Model", "")]


def legacy_metrics(report: dict | None) -> list[dict]:
    if not report:
        return []
    if report.get("metrics_legacy"):
        return report["metrics_legacy"]
    return [m for m in report.get("metrics", []) if "Optimized" not in m.get("Model", "")]


def sidebar(report):
    st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=72)
    st.sidebar.title("PGM Medical Diagnosis")
    st.sidebar.caption("Bayesian Networks · UCI Heart Disease")
    st.sidebar.caption(f"App version: `{APP_VERSION}`")

    primary = primary_metrics(report)
    if primary:
        row = primary[0]
        st.sidebar.markdown("### ★ Best model")
        st.sidebar.success(
            f"**{row['Model']}**\n\n"
            f"Acc {_metric_pct(row['Accuracy'])} · Prec {_metric_pct(row['Precision'])}\n\n"
            f"Rec {_metric_pct(row['Recall'])} · F1 {_metric_pct(row['F1'])} · AUC {_metric_pct(row['ROC-AUC'])}"
        )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Representation** — directed DAGs\n\n"
        "**Learning** — MLE + Chow-Liu tree\n\n"
        "**Inference** — VE & Belief Propagation"
    )
    st.sidebar.caption("Educational demo — not for clinical use.")


def run_model_comparison(
    models: dict,
    preset_name: str,
    custom_evidence: dict[str, str] | None,
    report: dict | None,
) -> pd.DataFrame:
    """Run all 4 BNs on matching clinical presets — each model gives its own P(Yes)."""
    rows = []
    for label, key, is_opt in MODEL_CATALOG:
        model = models[key]
        if preset_name != "— Custom —":
            presets = OPTIMIZED_PRESETS if is_opt else PRESETS
            evidence = presets.get(preset_name, {})
        else:
            evidence = custom_evidence or {}
        if not evidence:
            continue
        threshold = get_model_threshold(report, label)
        trace = predict_disease(model, evidence, method="ve")
        p_yes = disease_probability(trace)
        rows.append({
            "Model": label,
            "P(Heart Disease)": round(p_yes, 4),
            "Decision": classify_at_threshold(p_yes, threshold),
            "Threshold": threshold,
            "MAP label": map_label(trace),
            "Nodes": len(model.nodes()),
            "Directed edges": len(model.edges()),
        })
    return pd.DataFrame(rows)


def tab_diagnosis(models: dict, report):
    st.header("Interactive diagnosis")
    st.markdown(
        "Each **Bayesian Network** is a **directed** graph: arrows point from parent variables "
        "to children. The query is **P(Heart Disease | your clinical evidence)**."
    )

    col_cfg, col_ev, col_run = st.columns([1.1, 1.3, 0.9])

    with col_cfg:
        network = st.selectbox(
            "1 · Choose Bayesian Network",
            [m[0] for m in MODEL_CATALOG],
            help="Each model has a different DAG structure and CPTs — predictions will differ.",
        )
        model_key = next(k for lbl, k, _ in MODEL_CATALOG if lbl == network)
        model = models[model_key]

        preset = st.selectbox(
            "2 · Clinical preset",
            ["— Custom —", "Healthy adult (low risk)", "Classic angina (high risk)",
             "Middle-aged atypical presentation", "Elderly with vessel disease"],
        )
        if preset != "— Custom —":
            _, _, presets, _ = model_ui_config(network)
            st.session_state["active_preset"] = preset
            st.session_state["preset_evidence"] = presets[preset]

        if st.button("Load random test patient", width="stretch"):
            test = load_test_sample(optimized="Optimized" in network)
            row = test.sample(1, random_state=int(st.session_state.get("rand_seed", 0))).iloc[0]
            st.session_state["preset_evidence"] = records_to_evidence(
                row, feature_vars=[v for v in model.nodes() if v != TARGET]
            )
            st.session_state["rand_seed"] = st.session_state.get("rand_seed", 0) + 1
            st.session_state["true_label"] = row[TARGET]

        if "true_label" in st.session_state:
            st.info(f"Test-set ground truth: **{st.session_state['true_label']}**")

    with col_ev:
        st.subheader("3 · Patient evidence")
        evidence = collect_evidence_for_model(
            model,
            network,
            key_prefix="diag_",
            preset_evidence=st.session_state.get("preset_evidence"),
        )
        st.markdown("**Evidence entered (summary)**")
        st.dataframe(evidence_summary_table(evidence, network), width="stretch", hide_index=True)

    with col_run:
        st.subheader("4 · Inference")
        threshold = st.slider(
            "Decision threshold P(Yes)",
            0.05, 0.95, float(get_model_threshold(report, network)), 0.005,
        )
        compare_ve_bp = st.checkbox("Compare VE vs BP (same model)", value=True)
        run_one = st.button("Run inference", type="primary", width="stretch")
        run_all = st.button("Compare all 4 models", width="stretch")

    if run_all:
        st.subheader("Model comparison — same clinical preset")
        st.caption(
            "Legacy models (Manual Structure, Naive Bayes, Chow-Liu) use 13 UCI features. "
            "Optimized Clinical BN uses 10 Cleveland binary features — structures differ, so P(Yes) should differ."
        )
        cmp_df = run_model_comparison(
            models, preset, evidence if preset == "— Custom —" else None, report
        )
        if cmp_df.empty:
            st.warning("Choose a named preset to compare all four models.")
        else:
            st.dataframe(cmp_df, width="stretch", hide_index=True)
            st.bar_chart(cmp_df.set_index("Model")["P(Heart Disease)"])

    if run_one:
        if compare_ve_bp:
            ve_trace, bp_trace = compare_inference(model, evidence)
            traces = [("Variable Elimination", ve_trace), ("Belief Propagation", bp_trace)]
        else:
            traces = [("Variable Elimination", predict_disease(model, evidence, method="ve"))]

        st.markdown("---")
        st.markdown(f"**Model:** {network}")
        cols = st.columns(len(traces))
        for col, (algo_name, trace) in zip(cols, traces):
            p_yes = disease_probability(trace)
            decision = classify_at_threshold(p_yes, threshold)
            label, color = risk_band(p_yes)
            dcolor = "#e74c3c" if decision == "Yes" else "#27ae60"
            with col:
                st.markdown(f"#### {algo_name}")
                st.markdown(
                    f"<div style='border-left:5px solid {color};padding:12px;background:{color}18'>"
                    f"<h2 style='margin:0;color:{color}'>{p_yes:.1%}</h2>"
                    f"<p><b>{label}</b> · MAP: {map_label(trace)}</p>"
                    f"<p><b>Decision (t={threshold:.2f}):</b> "
                    f"<span style='color:{dcolor};font-weight:bold'>{decision}</span></p>"
                    f"<small>{trace.elapsed_ms:.1f} ms</small></div>",
                    unsafe_allow_html=True,
                )
                st.bar_chart(pd.DataFrame({
                    "Outcome": list(trace.posterior.keys()),
                    "P": list(trace.posterior.values()),
                }).set_index("Outcome"))

        if "true_label" in st.session_state:
            pred = classify_at_threshold(disease_probability(traces[0][1]), threshold)
            ok = pred == st.session_state["true_label"]
            (st.success if ok else st.warning)(
                f"Predicted **{pred}** vs actual **{st.session_state['true_label']}**"
            )

        with st.expander("Sensitivity analysis (what-if)"):
            sens = sensitivity_analysis(model, evidence)[:8]
            st.dataframe(pd.DataFrame(sens)[["variable", "from", "to", "p_yes", "delta"]], width="stretch")
    elif not run_all:
        st.info("Set patient evidence, then **Run inference** or **Compare all 4 models**.")


def tab_network_explorer(models: dict, report):
    st.header("Network explorer — directed DAGs")
    st.markdown("Orange arrows show **parent → child** direction (Representation pillar).")

    choice = st.selectbox("Model", [m[0] for m in MODEL_CATALOG])
    key = next(k for lbl, k, _ in MODEL_CATALOG if lbl == choice)
    model = models[key]

    col1, col2 = st.columns([1.2, 1])
    with col1:
        png = draw_directed_dag(model, f"{choice}")
        st.image(png, width="stretch")
    with col2:
        st.metric("Nodes", len(model.nodes()))
        st.metric("Directed edges", len(model.edges()))
        st.metric("Is DAG", "Yes" if nx.is_directed_acyclic_graph(model) else "No")
        st.markdown("**Edges (parent → child)**")
        st.dataframe(
            pd.DataFrame(list(model.edges()), columns=["Parent →", "Child"]),
            height=300,
            width="stretch",
        )

    if report:
        primary = primary_metrics(report)
        legacy = legacy_metrics(report)
        if primary:
            st.markdown("**Primary model metrics**")
            st.dataframe(pd.DataFrame(primary), width="stretch", hide_index=True)
        if legacy:
            st.markdown("**Multi-source baseline metrics**")
            st.dataframe(pd.DataFrame(legacy), width="stretch", hide_index=True)


def tab_algorithm_lab(models: dict):
    st.header("Algorithm laboratory")
    choice = st.selectbox("Model", [m[0] for m in MODEL_CATALOG], key="lab_model")
    key = next(k for lbl, k, _ in MODEL_CATALOG if lbl == choice)
    model = models[key]
    evidence = collect_evidence_for_model(model, choice, key_prefix="lab_")

    if st.button("Compare VE vs BP", type="primary"):
        ve, bp = compare_inference(model, evidence)
        df = pd.DataFrame([
            {"Algorithm": ve.method, "P(Yes)": disease_probability(ve), "MAP": map_label(ve), "ms": ve.elapsed_ms},
            {"Algorithm": bp.method, "P(Yes)": disease_probability(bp), "MAP": map_label(bp), "ms": bp.elapsed_ms},
        ])
        st.dataframe(df, width="stretch")
        diff = abs(disease_probability(ve) - disease_probability(bp))
        if diff < 1e-4:
            st.success(f"Algorithms agree (|Δ| = {diff:.2e}).")
        else:
            st.warning(f"Posterior difference |Δ| = {diff:.4f}")


def tab_metrics(report):
    st.header("Model performance")
    primary = primary_metrics(report)
    legacy = legacy_metrics(report)
    if primary:
        row = primary[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", _metric_pct(row["Accuracy"]))
        c2.metric("Precision", _metric_pct(row["Precision"]))
        c3.metric("Recall", _metric_pct(row["Recall"]))
        c4.metric("F1", _metric_pct(row["F1"]))
        c5.metric("ROC-AUC", _metric_pct(row["ROC-AUC"]))
        for fname, cap in [
            ("fig2_metrics.png", "Metrics"),
            ("fig3_confusion_best.png", "Confusion matrix"),
            ("fig4_roc_best.png", "ROC curve"),
        ]:
            p = OUTPUT_DIR / fname
            if p.exists():
                st.image(str(p), caption=cap, width="stretch")
    if legacy:
        st.subheader("Multi-source baselines")
        st.dataframe(pd.DataFrame(legacy), width="stretch", hide_index=True)


def tab_analysis_gallery(report):
    st.header("Analysis gallery")
    figures = [
        ("fig_eda_dashboard.png", "EDA — Cleveland"),
        ("fig0b_manual_layered_dag.png", "Manual Structure BN (directed DAG)"),
        ("fig_inference_scenarios.png", "Inference scenarios"),
        ("fig_cpt_analysis.png", "CPT analysis"),
        ("fig_mle_vs_bayesian.png", "MLE vs Bayesian"),
        ("fig5_inference_benchmark.png", "VE vs BP timing"),
    ]
    cols = st.columns(2)
    for i, (fname, cap) in enumerate(figures):
        p = OUTPUT_DIR / fname
        with cols[i % 2]:
            if p.exists():
                st.image(str(p), caption=cap, width="stretch")
            else:
                st.warning(f"Missing {fname} — run `python run.py`")


def tab_pgm_concepts():
    st.header("PGM guide")
    st.markdown("""
| Pillar | Demo |
|--------|------|
| **Representation** | Network Explorer — **directed** DAG, parent → child |
| **Learning** | Analysis Gallery — CPTs, MLE vs Bayesian |
| **Inference** | Diagnosis — P(Heart Disease \\| evidence), VE & BP |
| **Evaluation** | Metrics tab — ≥85% on Optimized Clinical BN |

**Recommended demo:** Preset *Classic angina (high risk)* → **Compare all 4 models** → show different P(Yes).
    """)


def main():
    report = load_report()
    sidebar(report)
    try:
        models = load_all_models()
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.stop()

    tabs = st.tabs(["Diagnosis", "Metrics", "Analysis Gallery", "Algorithm Lab", "Network Explorer", "PGM Guide"])
    with tabs[0]:
        tab_diagnosis(models, report)
    with tabs[1]:
        tab_metrics(report)
    with tabs[2]:
        tab_analysis_gallery(report)
    with tabs[3]:
        tab_algorithm_lab(models)
    with tabs[4]:
        tab_network_explorer(models, report)
    with tabs[5]:
        tab_pgm_concepts()


main()
