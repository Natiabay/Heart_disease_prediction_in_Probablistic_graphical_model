"""
Heart Disease Bayesian Network — Interactive PGM Demo
======================================================
Deploy: streamlit run app/streamlit_app.py
Streamlit Cloud main file: app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.presets import PRESETS
from src.config import FEATURE_LABELS, OUTPUT_DIR, STATE_LABELS, TARGET, ProjectConfig
from src.data import load_cached_or_build, prepare_train_test, records_to_evidence, train_test_split_data
from src.inference import (
    compare_inference,
    disease_probability,
    map_label,
    predict_disease,
    sensitivity_analysis,
)
from src.learning import (
    learn_expert_bn,
    learn_naive_bayes_bn,
    learn_structure_and_parameters,
)

st.set_page_config(
    page_title="Heart Disease BN | PGM Medical Diagnosis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPORT_PATH = OUTPUT_DIR / "report.json"


@st.cache_resource(show_spinner="Training Bayesian Networks on UCI data …")
def load_models():
    df = load_cached_or_build()
    train, _, _ = prepare_train_test(df)
    expert = learn_expert_bn(train)
    naive = learn_naive_bayes_bn(train)
    tree = learn_structure_and_parameters(train, ProjectConfig(structure_learning_iters=40))
    return expert.model, naive.model, tree.model


@st.cache_data
def load_report():
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return None


@st.cache_data
def load_test_sample():
    df = load_cached_or_build()
    _, test, _ = prepare_train_test(df)
    return test


def render_dag_figure(model, title: str) -> bytes:
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    pos = nx.spring_layout(G, seed=42, k=1.6)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#c0392b" if n == TARGET else "#2980b9" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#7f8c8d", arrows=True, arrowsize=16, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def risk_band(p: float) -> tuple[str, str]:
    if p < 0.25:
        return "Low risk", "#27ae60"
    if p < 0.55:
        return "Moderate risk", "#f39c12"
    return "High risk", "#e74c3c"


def sidebar(report):
    st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=72)
    st.sidebar.title("PGM Medical Diagnosis")
    st.sidebar.caption("Bayesian Networks · UCI Heart Disease · Educational demo")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Three PGM pillars")
    st.sidebar.info(
        "**Representation** — Expert + learned DAGs\n\n"
        "**Learning** — MLE/Bayesian CPTs, Hill Climb structure\n\n"
        "**Inference** — Variable Elimination & Belief Propagation"
    )

    if report:
        ds = report["dataset"]
        st.sidebar.markdown("### Dataset")
        st.sidebar.write(f"**{ds['n_samples']}** patients (4 UCI sources)")
        st.sidebar.write(f"Prevalence: **{ds['prevalence']:.1%}**")
        st.sidebar.json(ds["by_source"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "*Not for clinical use. Demonstrates probabilistic reasoning under uncertainty.*"
    )


def collect_evidence(model, key_prefix: str = "") -> dict[str, str]:
    """Build evidence from widgets; options match trained model states."""
    evidence = {}
    feature_vars = [v for v in STATE_LABELS if v != TARGET]
    cols = st.columns(2)
    for i, var in enumerate(feature_vars):
        model_states = []
        for cpd in model.get_cpds():
            if cpd.variable == var:
                model_states = [str(s) for s in cpd.state_names[var]]
                break
        options = model_states or STATE_LABELS[var]
        with cols[i % 2]:
            evidence[var] = st.selectbox(
                FEATURE_LABELS.get(var, var),
                options,
                key=f"{key_prefix}sel_{var}",
            )
    return evidence


def tab_diagnosis(expert_model, naive_model, tree_model, report):
    st.header("Interactive diagnosis")
    st.markdown(
        "Enter patient symptoms and risk factors. The BN computes "
        "**P(Heart Disease | evidence)** using exact probabilistic inference."
    )

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        network = st.selectbox(
            "Bayesian Network model",
            ["Naive Bayes BN (recommended)", "Chow-Liu Tree BN", "Expert BN"],
        )
        model_map = {
            "Naive Bayes BN (recommended)": naive_model,
            "Chow-Liu Tree BN": tree_model,
            "Expert BN": expert_model,
        }
        model = model_map[network]

        preset = st.selectbox("Load preset profile", ["— Custom —"] + list(PRESETS.keys()))
        if preset != "— Custom —":
            st.session_state["preset_evidence"] = PRESETS[preset]

        if st.button("Load random test patient", use_container_width=True):
            test = load_test_sample()
            row = test.sample(1, random_state=int(st.session_state.get("rand_seed", 0))).iloc[0]
            st.session_state["preset_evidence"] = records_to_evidence(row)
            st.session_state["rand_seed"] = st.session_state.get("rand_seed", 0) + 1
            st.session_state["true_label"] = row[TARGET]

        if "true_label" in st.session_state:
            st.caption(f"Hidden ground truth (test set): **{st.session_state['true_label']}**")

    with c2:
        st.subheader("Patient evidence")
        # Apply preset via session state defaults
        preset_ev = st.session_state.get("preset_evidence", {})
        feature_vars = [v for v in STATE_LABELS if v != TARGET]
        evidence = {}
        cols = st.columns(2)
        for i, var in enumerate(feature_vars):
            model_states = []
            for cpd in model.get_cpds():
                if cpd.variable == var:
                    model_states = [str(s) for s in cpd.state_names[var]]
                    break
            options = model_states or STATE_LABELS[var]
            default_idx = 0
            if var in preset_ev and preset_ev[var] in options:
                default_idx = options.index(preset_ev[var])
            with cols[i % 2]:
                evidence[var] = st.selectbox(
                    FEATURE_LABELS.get(var, var),
                    options,
                    index=default_idx,
                    key=f"diag_{var}",
                )

    with c3:
        st.subheader("Inference settings")
        compare_both = st.checkbox("Compare VE vs BP", value=True)
        run = st.button("Run Bayesian inference", type="primary", use_container_width=True)

    if run:
        if compare_both:
            ve_trace, bp_trace = compare_inference(model, evidence)
            traces = [ve_trace, bp_trace]
        else:
            method = st.radio("Algorithm", ["Variable Elimination", "Belief Propagation"], horizontal=True)
            inf = "ve" if "Elimination" in method else "bp"
            traces = [predict_disease(model, evidence, method=inf)]

        st.markdown("---")
        cols = st.columns(len(traces))
        for col, trace in zip(cols, traces):
            p_yes = disease_probability(trace)
            label, color = risk_band(p_yes)
            with col:
                st.markdown(f"#### {trace.method}")
                st.markdown(
                    f"<div style='background:{color}22;border-left:6px solid {color};"
                    f"padding:16px;border-radius:8px'>"
                    f"<h2 style='margin:0;color:{color}'>{p_yes:.1%}</h2>"
                    f"<p style='margin:4px 0 0 0'><b>{label}</b> · MAP: {map_label(trace)}</p>"
                    f"<p style='margin:4px 0 0 0;font-size:0.85em'>"
                    f"Inference: {trace.elapsed_ms:.2f} ms</p></div>",
                    unsafe_allow_html=True,
                )
                chart = pd.DataFrame({
                    "Outcome": list(trace.posterior.keys()),
                    "Probability": list(trace.posterior.values()),
                })
                st.bar_chart(chart.set_index("Outcome"))

        with st.expander("What-if sensitivity — which symptoms change the probability most?"):
            sens = sensitivity_analysis(model, evidence, method="ve")[:8]
            st.dataframe(
                pd.DataFrame(sens)[["variable", "from", "to", "p_yes", "delta"]],
                use_container_width=True,
            )
            st.caption(
                "Shows how P(Yes) changes when one evidence variable is flipped "
                "while others stay fixed — demonstrates causal/probabilistic sensitivity."
            )

        with st.expander("Inference trace (PGM — for instructor)"):
            for trace in traces:
                st.markdown(f"**{trace.method}**")
                for note in trace.notes:
                    st.write(f"- {note}")
                if trace.elimination_order:
                    st.write(f"Elimination order: `{trace.elimination_order}`")
                st.json(trace.posterior)
    else:
        st.info("Configure patient evidence and click **Run Bayesian inference**.")


def tab_algorithm_lab(expert_model, naive_model, tree_model):
    st.header("Algorithm laboratory")
    st.markdown(
        "Compare **Variable Elimination** and **Belief Propagation** on identical evidence. "
        "Both implement the Inference pillar; results should match on the same DAG."
    )

    model_choice = st.radio(
        "Model",
        ["Naive Bayes BN", "Chow-Liu Tree BN", "Expert BN"],
        horizontal=True,
    )
    model_map = {
        "Naive Bayes BN": naive_model,
        "Chow-Liu Tree BN": tree_model,
        "Expert BN": expert_model,
    }
    model = model_map[model_choice]
    evidence = collect_evidence(model, key_prefix="lab_")

    if st.button("Compare algorithms", type="primary"):
        ve, bp = compare_inference(model, evidence)
        df = pd.DataFrame([
            {
                "Algorithm": ve.method,
                "P(Yes)": disease_probability(ve),
                "MAP": map_label(ve),
                "Time (ms)": ve.elapsed_ms,
            },
            {
                "Algorithm": bp.method,
                "P(Yes)": disease_probability(bp),
                "MAP": map_label(bp),
                "Time (ms)": bp.elapsed_ms,
            },
        ])
        st.dataframe(df, use_container_width=True)
        diff = abs(disease_probability(ve) - disease_probability(bp))
        if diff < 1e-4:
            st.success(f"Posteriors agree (|Δ| = {diff:.2e}) — both algorithms are consistent.")
        else:
            st.warning(f"Posterior difference |Δ| = {diff:.4f} (may occur on approximate BP).")


def tab_network_explorer(expert_model, naive_model, tree_model, report):
    st.header("Network explorer")
    st.markdown("Visualize the **Representation** pillar: DAG structure learned vs expert-defined.")

    choice = st.radio(
        "View network",
        ["Naive Bayes BN", "Chow-Liu Tree BN", "Expert BN"],
        horizontal=True,
    )
    model_map = {
        "Naive Bayes BN": naive_model,
        "Chow-Liu Tree BN": tree_model,
        "Expert BN": expert_model,
    }
    model = model_map[choice]

    col1, col2 = st.columns([1.2, 1])
    with col1:
        png = render_dag_figure(model, f"{choice} — Heart Disease DAG")
        st.image(png, use_container_width=True)
    with col2:
        st.metric("Nodes", len(model.nodes()))
        st.metric("Edges", len(model.edges()))
        st.metric("Is DAG", "Yes" if nx.is_directed_acyclic_graph(model) else "No")
        st.markdown("**Edges (parent → child)**")
        st.dataframe(
            pd.DataFrame(list(model.edges()), columns=["Parent", "Child"]),
            height=320,
            use_container_width=True,
        )

    if report and "metrics" in report:
        st.subheader("Evaluation metrics (Learning + Inference)")
        st.dataframe(pd.DataFrame(report["metrics"]), use_container_width=True)

        metrics_path = OUTPUT_DIR / "metrics.csv"
        if metrics_path.exists():
            st.download_button(
                "Download metrics CSV",
                metrics_path.read_bytes(),
                file_name="heart_disease_bn_metrics.csv",
            )


def tab_pgm_concepts():
    st.header("PGM concepts — course demo guide")
    st.markdown("""
### Project: Medical Diagnosis System Using Bayesian Networks

This system implements your course proposal on **heart disease prediction**
using publicly available UCI data.

| Pillar | What to show the instructor |
|--------|----------------------------|
| **Representation** | Tab *Network Explorer* — DAG with symptoms → disease |
| **Learning** | Expert CPTs (Bayesian) + data-driven structure (Hill Climb + BIC) |
| **Inference** | Tab *Algorithm Lab* — VE vs BP on same evidence |
| **Uncertainty** | Diagnosis tab — probability bar, not just yes/no |
| **Evaluation** | Metrics table — accuracy, precision, recall, F1, ROC-AUC |

### Key query

> **P(Heart Disease | chest pain, BP, cholesterol, …)**

### Why Bayesian Networks?

- Handles **partial evidence** (missing tests → Unknown state)
- Outputs **calibrated probabilities**, not black-box scores
- Graph is **interpretable** — each edge is a conditional dependency

### Dataset

[UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease) —
Cleveland, Hungarian, Switzerland, VA Long Beach (~920 patients).

### Run locally

```bash
pip install -r requirements.txt
python run.py
streamlit run app/streamlit_app.py
```
    """)


def main():
    report = load_report()
    sidebar(report)
    expert_model, naive_model, tree_model = load_models()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Diagnosis",
        "Algorithm Lab",
        "Network Explorer",
        "PGM Guide",
    ])

    with tab1:
        tab_diagnosis(expert_model, naive_model, tree_model, report)
    with tab2:
        tab_algorithm_lab(expert_model, naive_model, tree_model)
    with tab3:
        tab_network_explorer(expert_model, naive_model, tree_model, report)
    with tab4:
        tab_pgm_concepts()


if __name__ == "__main__":
    main()
