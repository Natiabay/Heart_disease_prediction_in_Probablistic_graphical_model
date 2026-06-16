#!/usr/bin/env python3
"""
Medical Diagnosis System Using Bayesian Networks for Heart Disease Prediction
==============================================================================

PGM course project covering:
  - Representation: expert DAG + data-driven DAG
  - Learning: MLE / Bayesian CPT fitting, structure learning (Hill Climb + BIC)
  - Inference: Variable Elimination and Belief Propagation

Usage:
    python run.py
    python run.py --quick
    python run.py --query --cp "Typical angina" --exang Yes
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from src.artifacts import save_model
from src.config import OUTPUT_DIR, PROJECT_ROOT, STATE_LABELS, TARGET, ProjectConfig
from src.data import (
    dataset_summary,
    load_discretized_dataset,
    records_to_evidence,
    train_test_split_data,
)
from src.evaluation import benchmark_inference_methods, evaluate_model, results_to_dataframe
from src.inference import predict_disease
from src.learning import get_learning_summary, learn_expert_bn, learn_structure_and_parameters
from src.model import build_expert_structure, representation_summary
from src.visualization import (
    plot_confusion_matrix,
    plot_dag,
    plot_inference_benchmark,
    plot_metrics_comparison,
    plot_roc_curve,
)


def run_pipeline(output_dir: Path, quick: bool = False, use_hillclimb: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        quick=quick,
        structure_learning_iters=25 if quick else 40,
        structure_sample_size=250 if quick else 400,
    )

    print("=" * 64)
    print("PGM Project: Heart Disease Bayesian Network")
    print("=" * 64)

    # --- Data ---
    print("\n[Data] Loading UCI Heart Disease (multi-source)")
    df = load_discretized_dataset()
    summary = dataset_summary(df)
    print(f"  Samples: {summary['n_samples']}  |  Sources: {summary['by_source']}")
    print(f"  Heart disease prevalence: {summary['prevalence']:.1%}")

    train_df, test_df = train_test_split_data(df, test_size=config.test_size, seed=config.seed)
    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")

    # --- Phase 1: Representation ---
    print("\n[Phase 1] Representation — Bayesian Network structure")
    expert_meta = build_expert_structure()
    print(f"  Expert BN: {expert_meta['name']}")
    print(f"  Variables: {len(expert_meta['variables'])}")
    print(f"  Edges: {len(expert_meta['edges'])}")
    print(f"  {expert_meta['description']}")

    # --- Phase 2: Learning ---
    print("\n[Phase 2] Learning — parameter & structure learning")
    expert_learned = learn_expert_bn(train_df, use_bayesian=True)
    learned = learn_structure_and_parameters(train_df, config, use_hillclimb=use_hillclimb)

    for res in (expert_learned, learned):
        s = get_learning_summary(res)
        print(f"  {s['name']}: {s['method']}")
        print(f"    Nodes={s['n_nodes']}, Edges={s['n_edges']}", end="")
        if s["score"] is not None:
            print(f", BIC={s['score']:.1f}", end="")
        print()

    plot_dag(
        expert_learned.model,
        output_dir / "fig0_expert_dag.png",
        "Expert Bayesian Network (Heart Disease)",
    )
    plot_dag(
        learned.model,
        output_dir / "fig1_learned_dag.png",
        "Data-Driven Bayesian Network (Chow-Liu / Hill Climb)",
    )

    # --- Phase 3: Inference + Evaluation ---
    print("\n[Phase 3] Inference — Variable Elimination & Belief Propagation")
    results = []
    eval_df = test_df.head(100) if quick else test_df
    bp_sample = eval_df.head(25)

    for lr in (expert_learned, learned):
        ev_ve = evaluate_model(lr.model, eval_df, lr.name, method="ve")
        results.append(ev_ve)
        print(
            f"  {ev_ve.model_name} [VE]: "
            f"acc={ev_ve.accuracy:.3f} prec={ev_ve.precision:.3f} "
            f"rec={ev_ve.recall:.3f} f1={ev_ve.f1:.3f} "
            f"auc={ev_ve.roc_auc:.3f} ({ev_ve.mean_inference_ms:.2f} ms/query)",
            flush=True,
        )
        ev_bp = evaluate_model(lr.model, bp_sample, lr.name, method="bp")
        ev_bp.n_test = len(bp_sample)
        results.append(ev_bp)
        print(
            f"  {ev_bp.model_name} [BP]: "
            f"acc={ev_bp.accuracy:.3f} prec={ev_bp.precision:.3f} "
            f"rec={ev_bp.recall:.3f} f1={ev_bp.f1:.3f} "
            f"auc={ev_bp.roc_auc:.3f} ({ev_bp.mean_inference_ms:.2f} ms/query, n={len(bp_sample)})",
            flush=True,
        )

    print("\n[Phase 3] Saving figures …", flush=True)

    metrics_df = results_to_dataframe(results)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    plot_metrics_comparison(metrics_df, output_dir / "fig2_metrics.png")

    # Best expert VE result for detailed plots
    expert_ve = next(r for r in results if "Expert" in r.model_name and r.method == "VE")
    plot_confusion_matrix(expert_ve, output_dir / "fig3_confusion_expert_ve.png")
    plot_roc_curve(expert_ve, output_dir / "fig4_roc_expert_ve.png")

    bench = benchmark_inference_methods(
        expert_learned.model,
        test_df.head(15),
        expert_learned.name,
        n_repeats=1,
    )
    bench.to_csv(output_dir / "inference_benchmark.csv", index=False)
    plot_inference_benchmark(bench, output_dir / "fig5_inference_benchmark.png")

    # Example query
    example = test_df.iloc[0]
    evidence = records_to_evidence(example)
    trace = predict_disease(expert_learned.model, evidence, method="ve")
    print("\n[Demo query]")
    print(f"  Evidence (sample patient): {evidence}")
    print(f"  P(HeartDisease=Yes | evidence) = {trace.posterior.get('Yes', 0):.3f}")
    print(f"  True label: {example[TARGET]}")
    print(f"  Inference: {trace.method} in {trace.elapsed_ms:.2f} ms")

    # --- Save artifacts for Streamlit ---
    print("\n[Deploy] Saving model artifacts")
    save_model(expert_learned.model, "expert_heart_disease_bn", get_learning_summary(expert_learned))
    save_model(learned.model, "data_driven_heart_disease_bn", get_learning_summary(learned))

    report = {
        "dataset": summary,
        "expert_structure": expert_meta,
        "expert_repr": representation_summary(expert_learned.model),
        "learned_repr": representation_summary(learned.model),
        "metrics": metrics_df.to_dict(orient="records"),
        "state_labels": STATE_LABELS,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"Done. Figures and metrics → {output_dir}")
    print("Streamlit demo: streamlit run app/streamlit_app.py")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart Disease BN — PGM pipeline")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true", help="Faster evaluation settings")
    parser.add_argument(
        "--hillclimb", action="store_true",
        help="Use Hill Climb structure learning (slower; default is Chow-Liu tree)",
    )
    args = parser.parse_args()
    run_pipeline(args.output, quick=args.quick, use_hillclimb=args.hillclimb)


if __name__ == "__main__":
    main()
