#!/usr/bin/env python3
"""
Medical Diagnosis System Using Bayesian Networks for Heart Disease Prediction
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from src.artifacts import save_model
from src.config import OUTPUT_DIR, STATE_LABELS, TARGET, ProjectConfig
from src.data import (
    dataset_summary,
    load_discretized_dataset,
    prepare_train_test,
    records_to_evidence,
)
from src.evaluation import benchmark_inference_methods, evaluate_model, results_to_dataframe
from src.inference import predict_disease
from src.learning import (
    get_learning_summary,
    learn_expert_bn,
    learn_naive_bayes_bn,
    learn_structure_and_parameters,
)
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
    config = ProjectConfig(quick=quick)

    print("=" * 64)
    print("PGM Project: Heart Disease Bayesian Network")
    print("=" * 64)

    print("\n[Data] Loading UCI Heart Disease (multi-source)")
    df = load_discretized_dataset()
    summary = dataset_summary(df)
    print(f"  Samples: {summary['n_samples']}  |  Sources: {summary['by_source']}")
    print(f"  Heart disease prevalence: {summary['prevalence']:.1%}")

    train_df, test_df, _ = prepare_train_test(df, test_size=config.test_size, seed=config.seed)
    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)} (Unknown imputed from train modes)")

    print("\n[Phase 1] Representation — three Bayesian Network structures")
    expert_meta = build_expert_structure()
    print(f"  1. Expert BN — {len(expert_meta['edges'])} edges (clinical intuition)")
    print(f"  2. Naive Bayes BN — {len([v for v in expert_meta['variables'] if v != TARGET])} edges (symptoms → disease)")
    print(f"  3. Chow-Liu Tree BN — learned from data (mutual information)")

    print("\n[Phase 2] Learning — parameter learning on all structures")
    expert_learned = learn_expert_bn(train_df, use_bayesian=True)
    naive_learned = learn_naive_bayes_bn(train_df)
    tree_learned = learn_structure_and_parameters(train_df, config, use_hillclimb=use_hillclimb)

    models = [expert_learned, naive_learned, tree_learned]
    for res in models:
        s = get_learning_summary(res)
        print(f"  {s['name']}: {s['method']}")
        print(f"    Nodes={s['n_nodes']}, Edges={s['n_edges']}", end="")
        if s["score"] is not None:
            print(f", BIC={s['score']:.1f}", end="")
        print()

    plot_dag(expert_learned.model, output_dir / "fig0_expert_dag.png", "Expert Bayesian Network")
    plot_dag(naive_learned.model, output_dir / "fig1_naive_bayes_dag.png", "Naive Bayes Diagnosis BN")
    plot_dag(tree_learned.model, output_dir / "fig1b_chowliu_dag.png", "Chow-Liu Tree BN")

    print("\n[Phase 3] Inference — VE with tuned decision threshold (trained on train set)")
    results = []
    eval_df = test_df.head(100) if quick else test_df

    for lr in models:
        ev = evaluate_model(
            lr.model,
            eval_df,
            lr.name,
            method="ve",
            tune_threshold_df=train_df,
        )
        results.append(ev)
        print(
            f"  {ev.model_name} [VE, t={ev.threshold:.2f}]: "
            f"acc={ev.accuracy:.3f} prec={ev.precision:.3f} "
            f"rec={ev.recall:.3f} f1={ev.f1:.3f} auc={ev.roc_auc:.3f} "
            f"({ev.mean_inference_ms:.1f} ms/query)",
            flush=True,
        )

    print("\n[Phase 3] Saving figures …", flush=True)
    metrics_df = results_to_dataframe(results)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    plot_metrics_comparison(metrics_df, output_dir / "fig2_metrics.png")

    best = max(results, key=lambda r: r.f1)
    print(f"\n  Best model by F1: {best.model_name} (F1={best.f1:.3f}, acc={best.accuracy:.3f})")
    plot_confusion_matrix(best, output_dir / "fig3_confusion_best.png")
    plot_roc_curve(best, output_dir / "fig4_roc_best.png")

    bench = benchmark_inference_methods(naive_learned.model, test_df.head(15), naive_learned.name)
    bench.to_csv(output_dir / "inference_benchmark.csv", index=False)
    plot_inference_benchmark(bench, output_dir / "fig5_inference_benchmark.png")

    example = test_df.iloc[0]
    evidence = records_to_evidence(example)
    trace = predict_disease(naive_learned.model, evidence, method="ve")
    print("\n[Demo query — Naive Bayes BN]")
    print(f"  P(HeartDisease=Yes | evidence) = {trace.posterior.get('Yes', 0):.3f}")
    print(f"  True label: {example[TARGET]}")

    print("\n[Deploy] Saving model artifacts")
    for lr in models:
        slug = lr.name.lower().replace(" ", "_")
        save_model(lr.model, slug, get_learning_summary(lr))

    report = {
        "dataset": summary,
        "expert_structure": expert_meta,
        "best_model": best.model_name,
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
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--hillclimb", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.output, quick=args.quick, use_hillclimb=args.hillclimb)


if __name__ == "__main__":
    main()
