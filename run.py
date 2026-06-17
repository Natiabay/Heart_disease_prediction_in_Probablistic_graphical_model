#!/usr/bin/env python3
"""
Medical Diagnosis System Using Bayesian Networks for Heart Disease Prediction
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

from src.artifacts import save_model
from src.config import (
    OPTIMIZED_SEED,
    OPTIMIZED_VARS,
    OUTPUT_DIR,
    STATE_LABELS,
    TARGET,
    ProjectConfig,
)
from src.data import (
    dataset_summary,
    load_discretized_dataset,
    load_optimized_dataset,
    prepare_train_test,
    prepare_train_val_test,
    records_to_evidence,
)
from src.evaluation import benchmark_inference_methods, evaluate_model, results_to_dataframe
from src.inference import predict_disease
from src.learning import (
    get_learning_summary,
    learn_manual_structure_bn,
    learn_naive_bayes_bn,
    learn_optimized_clinical_bn,
    learn_structure_and_parameters,
)
from src.model import build_manual_structure
from src.eda import plot_eda_dashboard
from src.visualization import (
    plot_confusion_matrix,
    plot_cpt_analysis,
    plot_dag,
    plot_inference_benchmark,
    plot_inference_scenarios,
    plot_layered_dag,
    plot_metrics_comparison,
    plot_mle_vs_bayesian_evaluation,
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

    print("\n[Data] Loading optimized Cleveland clinical subset")
    opt_df = load_optimized_dataset()
    opt_summary = dataset_summary(opt_df)
    opt_train, opt_val, opt_test = prepare_train_val_test(
        opt_df, test_size=config.test_size, seed=OPTIMIZED_SEED
    )
    opt_tune = opt_train if quick else pd.concat([opt_train, opt_val])
    print(
        f"  Cleveland samples: {opt_summary['n_samples']}  |  "
        f"Train/Val/Test: {len(opt_train)}/{len(opt_val)}/{len(opt_test)}"
    )

    print("\n[Phase 1] Representation — Bayesian Network structures")
    manual_meta = build_manual_structure()
    print(f"  1. Manual Structure BN — {len(manual_meta['edges'])} edges (hand-built DAG)")
    print(f"  2. Naive Bayes BN — symptoms → disease")
    print(f"  3. Chow-Liu Tree BN — learned from multi-source data")
    print(f"  4. Optimized Clinical BN — Cleveland binary features (primary)")

    print("\n[Phase 2] Learning — parameter learning on all structures")
    manual_learned = learn_manual_structure_bn(train_df, use_bayesian=True)
    naive_learned = learn_naive_bayes_bn(train_df)
    tree_learned = learn_structure_and_parameters(train_df, config, use_hillclimb=use_hillclimb)
    optimized_learned = learn_optimized_clinical_bn(opt_train)

    models = [manual_learned, naive_learned, tree_learned, optimized_learned]
    for res in models:
        s = get_learning_summary(res)
        print(f"  {s['name']}: {s['method']}")
        print(f"    Nodes={s['n_nodes']}, Edges={s['n_edges']}", end="")
        if s["score"] is not None:
            print(f", BIC={s['score']:.1f}", end="")
        print()

    plot_dag(manual_learned.model, output_dir / "fig0_manual_structure_dag.png", "Manual Structure BN")
    plot_dag(naive_learned.model, output_dir / "fig1_naive_bayes_dag.png", "Naive Bayes Diagnosis BN")
    plot_dag(tree_learned.model, output_dir / "fig1b_chowliu_dag.png", "Chow-Liu Tree BN")
    plot_dag(
        optimized_learned.model,
        output_dir / "fig1c_optimized_dag.png",
        "Optimized Clinical BN (Cleveland)",
    )
    plot_layered_dag(manual_learned.model, output_dir / "fig0b_manual_layered_dag.png")

    opt_features = [v for v in OPTIMIZED_VARS if v != TARGET]
    eval_df = test_df.head(100) if quick else test_df
    opt_eval_df = opt_test.head(50) if quick else opt_test

    print("\n[Analysis] EDA, CPT, inference scenarios, MLE vs Bayesian …")
    plot_eda_dashboard(output_dir / "fig_eda_dashboard.png")
    plot_cpt_analysis(manual_learned.model, output_dir / "fig_cpt_analysis.png")
    plot_inference_scenarios(
        manual_learned.model,
        output_dir / "fig_inference_scenarios.png",
        threshold=0.5,
    )

    manual_mle = learn_manual_structure_bn(train_df, use_bayesian=False)
    mle_ev = evaluate_model(manual_mle.model, eval_df, "Manual Structure BN (MLE)", tune_threshold_df=train_df)
    bayes_ev = evaluate_model(
        manual_learned.model, eval_df, "Manual Structure BN (Bayesian)", tune_threshold_df=train_df
    )
    plot_mle_vs_bayesian_evaluation(mle_ev, bayes_ev, output_dir / "fig_mle_vs_bayesian.png")

    print("\n[Phase 3] Inference — VE with tuned decision thresholds")
    legacy_results: list = []
    primary_results: list = []

    print("  Multi-source models (920 patients, full discretized features):")
    for lr in [manual_learned, naive_learned, tree_learned]:
        ev = evaluate_model(
            lr.model,
            eval_df,
            lr.name,
            method="ve",
            tune_threshold_df=train_df,
        )
        legacy_results.append(ev)
        print(
            f"    {ev.model_name} [VE, t={ev.threshold:.2f}]: "
            f"acc={ev.accuracy:.3f} prec={ev.precision:.3f} "
            f"rec={ev.recall:.3f} f1={ev.f1:.3f} auc={ev.roc_auc:.3f} "
            f"({ev.mean_inference_ms:.1f} ms/query)",
            flush=True,
        )

    print("  Cleveland optimized model (303 patients, clinical binary features):")
    opt_ev = evaluate_model(
        optimized_learned.model,
        opt_eval_df,
        optimized_learned.name,
        method="ve",
        tune_threshold_df=opt_tune,
        feature_vars=opt_features,
        balanced_threshold=True,
    )
    primary_results.append(opt_ev)
    print(
        f"    {opt_ev.model_name} [VE, t={opt_ev.threshold:.2f}] ★ PRIMARY: "
        f"acc={opt_ev.accuracy:.3f} prec={opt_ev.precision:.3f} "
        f"rec={opt_ev.recall:.3f} f1={opt_ev.f1:.3f} auc={opt_ev.roc_auc:.3f} "
        f"({opt_ev.mean_inference_ms:.1f} ms/query)",
        flush=True,
    )

    print("\n[Phase 3] Saving figures …", flush=True)
    legacy_df = results_to_dataframe(legacy_results, evaluation_set="Multi-source (920)")
    primary_df = results_to_dataframe(primary_results, evaluation_set="Cleveland optimized (303)")
    metrics_df = pd.concat([primary_df, legacy_df], ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    primary_df.to_csv(output_dir / "metrics_primary.csv", index=False)
    plot_metrics_comparison(primary_df, output_dir / "fig2_metrics.png")

    best = max(primary_results, key=lambda r: min(r.accuracy, r.precision, r.recall, r.f1, r.roc_auc))
    print(
        f"\n  Best model (max min-metric): {best.model_name} "
        f"(acc={best.accuracy:.3f}, prec={best.precision:.3f}, rec={best.recall:.3f}, "
        f"f1={best.f1:.3f}, auc={best.roc_auc:.3f})"
    )
    plot_confusion_matrix(best, output_dir / "fig3_confusion_best.png")
    plot_roc_curve(best, output_dir / "fig4_roc_best.png")

    bench = benchmark_inference_methods(
        optimized_learned.model,
        opt_eval_df.head(15),
        optimized_learned.name,
        feature_vars=opt_features,
    )
    bench.to_csv(output_dir / "inference_benchmark.csv", index=False)
    plot_inference_benchmark(bench, output_dir / "fig5_inference_benchmark.png")

    example = opt_test.iloc[0]
    evidence = records_to_evidence(example, feature_vars=opt_features)
    trace = predict_disease(optimized_learned.model, evidence, method="ve")
    print("\n[Demo query — Optimized Clinical BN]")
    print(f"  P(HeartDisease=Yes | evidence) = {trace.posterior.get('Yes', 0):.3f}")
    print(f"  True label: {example[TARGET]}")

    print("\n[Deploy] Saving model artifacts")
    for lr in models:
        slug = lr.name.lower().replace(" ", "_")
        save_model(lr.model, slug, get_learning_summary(lr))

    report = {
        "dataset": summary,
        "optimized_dataset": opt_summary,
        "manual_structure": manual_meta,
        "best_model": best.model_name,
        "optimized_threshold": opt_ev.threshold,
        "metrics": metrics_df.to_dict(orient="records"),
        "metrics_primary": primary_df.to_dict(orient="records"),
        "metrics_legacy": legacy_df.to_dict(orient="records"),
        "metrics_note": (
            "Primary model (Optimized Clinical BN) is evaluated on Cleveland hold-out "
            "with balanced threshold tuning on train+validation. Manual Structure and Naive Bayes "
            "use multi-source UCI data with wider discretization — lower accuracy but "
            "strong recall; they illustrate PGM representation, not peak ML performance."
        ),
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
