#!/usr/bin/env python3
"""
Advanced MNIST+MLP experiments:
1. few-shot recovery
2. extreme non-IID
3. gap decomposition
4. layer-wise ablation
5. weighting ablation
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from core_merge_lab import (
    ExperimentConfig,
    SmallMLP,
    build_datasets,
    clone_model,
    collect_regmean_stats,
    compute_client_validation_losses,
    config_to_jsonable,
    evaluate_merged_model,
    evaluate_model,
    fit_bias_only_calibration,
    fit_linear_probe,
    fit_ridge_probe,
    make_calibration_loader,
    make_class_balanced_subset,
    post_merge_finetune,
    regmean_merge,
    regmean_merge_selected_layers,
    run_single_experiment,
    save_json,
    set_seed,
    simple_average,
    train_centralized,
    train_clients,
    validation_loss_weights,
    weighted_average,
)
from report_utils_v2 import aggregate_rows, build_report_markdown, flatten_summary_row, write_csv


GROUP_NOTES = {
    "few_shot_recovery": "Post-merge few-shot calibration with more stable bias-only and ridge-style frozen-feature calibration.",
    "extreme_non_iid": "Pathological label-shard experiments to test whether RegMean gains grow under extreme non-IID.",
    "gap_decomposition": "Oracle-style head refit and short fine-tuning to separate merge gap from representation and optimization gap.",
    "layerwise_ablation": "Compare all-layer RegMean with selective late-layer RegMean.",
    "weighting_ablation": "Compare equal, epoch, sample, and validation-loss weighting strategies for both averaging and RegMean.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced MNIST merge experiments")
    parser.add_argument("--data-root", type=Path, default=Path("./work/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/mnist_advanced_suite"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--regmean-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=[
            "few_shot_recovery",
            "extreme_non_iid",
            "gap_decomposition",
            "layerwise_ablation",
            "weighting_ablation",
        ],
    )
    return parser.parse_args()


def make_base_config(args: argparse.Namespace, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        data_root=args.data_root,
        device=args.device,
        seed=seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        regmean_lambda=args.regmean_lambda,
    )


def prepare_core_objects(config: ExperimentConfig):
    set_seed(config.seed)
    train_dataset, test_dataset = build_datasets(config.data_root)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)
    base_model = SmallMLP(hidden_sizes=config.hidden_sizes)
    client_results = train_clients(config, train_dataset, base_model)
    client_models = [client.model for client in client_results]
    sample_weights = [len(client.spec.indices) for client in client_results]
    epoch_weights = [client.spec.epochs for client in client_results]
    val_losses = compute_client_validation_losses(client_results, train_dataset, config)
    val_weights = validation_loss_weights(val_losses)
    stats_per_client = [
        collect_regmean_stats(
            client.model,
            make_calibration_loader(
                train_dataset,
                client.spec.indices,
                config.calibration_size,
                config.eval_batch_size,
                config.seed + client.spec.client_id,
            ),
            torch.device(config.device),
        )
        for client in client_results
    ]
    centralized = train_centralized(config, train_dataset, base_model)
    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "test_loader": test_loader,
        "base_model": base_model,
        "client_results": client_results,
        "client_models": client_models,
        "sample_weights": sample_weights,
        "epoch_weights": epoch_weights,
        "val_weights": val_weights,
        "stats_per_client": stats_per_client,
        "centralized": centralized,
    }


def few_shot_recovery_results(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_core_objects(config)
    train_dataset = core["train_dataset"]
    test_loader = core["test_loader"]
    client_results = core["client_results"]
    client_models = core["client_models"]
    stats_per_client = core["stats_per_client"]

    simple = simple_average(client_models)
    regmean = regmean_merge(client_models, stats_per_client, [1.0] * len(client_models), config.regmean_lambda)
    results = {
        "simple_avg": evaluate_merged_model(simple, client_results, train_dataset, test_loader, config),
        "regmean_equal": evaluate_merged_model(regmean, client_results, train_dataset, test_loader, config),
        "centralized": evaluate_merged_model(core["centralized"], client_results, train_dataset, test_loader, config),
    }

    for shots in [1, 5, 10]:
        fewshot_indices = make_class_balanced_subset(train_dataset, shots, config.seed + shots)
        simple_bias = fit_bias_only_calibration(simple, train_dataset, fewshot_indices, config, steps=40, lr=3e-3)
        regmean_bias = fit_bias_only_calibration(regmean, train_dataset, fewshot_indices, config, steps=40, lr=3e-3)
        simple_ridge = fit_ridge_probe(simple, train_dataset, fewshot_indices, config, ridge_lambda=5.0)
        regmean_ridge = fit_ridge_probe(regmean, train_dataset, fewshot_indices, config, ridge_lambda=5.0)
        simple_tune = post_merge_finetune(simple, train_dataset, fewshot_indices, config, steps=40, lr=1e-4, layers_to_train=[2])
        regmean_tune = post_merge_finetune(regmean, train_dataset, fewshot_indices, config, steps=40, lr=1e-4, layers_to_train=[2])
        results[f"simple_bias_{shots}shot"] = evaluate_merged_model(simple_bias, client_results, train_dataset, test_loader, config)
        results[f"regmean_bias_{shots}shot"] = evaluate_merged_model(regmean_bias, client_results, train_dataset, test_loader, config)
        results[f"simple_ridge_{shots}shot"] = evaluate_merged_model(simple_ridge, client_results, train_dataset, test_loader, config)
        results[f"regmean_ridge_{shots}shot"] = evaluate_merged_model(regmean_ridge, client_results, train_dataset, test_loader, config)
        results[f"simple_lastlayer_tune_{shots}shot"] = evaluate_merged_model(simple_tune, client_results, train_dataset, test_loader, config)
        results[f"regmean_lastlayer_tune_{shots}shot"] = evaluate_merged_model(regmean_tune, client_results, train_dataset, test_loader, config)
    return {"config": config_to_jsonable(config), "merges": results}


def extreme_non_iid_results(config: ExperimentConfig) -> Dict[str, object]:
    config = deepcopy(config)
    config.split_mode = "pathological"
    config.pathological_classes_per_client = 2
    return run_single_experiment(config)


def gap_decomposition_results(config: ExperimentConfig) -> Dict[str, object]:
    config = deepcopy(config)
    config.split_mode = "dirichlet"
    config.dirichlet_alpha = 0.5
    core = prepare_core_objects(config)
    train_dataset = core["train_dataset"]
    test_loader = core["test_loader"]
    client_results = core["client_results"]
    client_models = core["client_models"]
    stats_per_client = core["stats_per_client"]
    merged = {
        "simple_avg": simple_average(client_models),
        "regmean_equal": regmean_merge(client_models, stats_per_client, [1.0] * len(client_models), config.regmean_lambda),
        "centralized": core["centralized"],
    }
    fewshot_indices = make_class_balanced_subset(train_dataset, 10, config.seed + 111)
    oracle_indices = make_class_balanced_subset(train_dataset, 50, config.seed + 222)
    outputs = {}
    for name, model in merged.items():
        outputs[name] = evaluate_merged_model(model, client_results, train_dataset, test_loader, config)
        if name == "centralized":
            continue
        head_refit = fit_linear_probe(model, train_dataset, fewshot_indices, config, steps=80, lr=1e-2)
        short_tune = post_merge_finetune(model, train_dataset, oracle_indices, config, steps=150, lr=5e-4)
        outputs[f"{name}_head_refit"] = evaluate_merged_model(head_refit, client_results, train_dataset, test_loader, config)
        outputs[f"{name}_short_finetune"] = evaluate_merged_model(short_tune, client_results, train_dataset, test_loader, config)
    return {"config": config_to_jsonable(config), "merges": outputs}


def layerwise_ablation_results(config: ExperimentConfig) -> Dict[str, object]:
    config = deepcopy(config)
    config.split_mode = "dirichlet"
    config.dirichlet_alpha = 0.5
    core = prepare_core_objects(config)
    train_dataset = core["train_dataset"]
    test_loader = core["test_loader"]
    client_results = core["client_results"]
    client_models = core["client_models"]
    stats_per_client = core["stats_per_client"]
    sample_weights = core["sample_weights"]
    outputs = {
        "simple_avg": evaluate_merged_model(simple_average(client_models), client_results, train_dataset, test_loader, config),
        "regmean_all_layers": evaluate_merged_model(regmean_merge(client_models, stats_per_client, [1.0] * len(client_models), config.regmean_lambda), client_results, train_dataset, test_loader, config),
        "regmean_last_layer": evaluate_merged_model(regmean_merge_selected_layers(client_models, stats_per_client, [1.0] * len(client_models), config.regmean_lambda, [2]), client_results, train_dataset, test_loader, config),
        "regmean_hidden_layers": evaluate_merged_model(regmean_merge_selected_layers(client_models, stats_per_client, [1.0] * len(client_models), config.regmean_lambda, [0, 1]), client_results, train_dataset, test_loader, config),
        "sample_weighted_last_layer": evaluate_merged_model(regmean_merge_selected_layers(client_models, stats_per_client, sample_weights, config.regmean_lambda, [2]), client_results, train_dataset, test_loader, config),
    }
    return {"config": config_to_jsonable(config), "merges": outputs}


def weighting_ablation_results(config: ExperimentConfig) -> Dict[str, object]:
    config = deepcopy(config)
    config.split_mode = "dirichlet"
    config.dirichlet_alpha = 0.5
    config.client_epochs = tuple([1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
    core = prepare_core_objects(config)
    train_dataset = core["train_dataset"]
    test_loader = core["test_loader"]
    client_results = core["client_results"]
    client_models = core["client_models"]
    stats_per_client = core["stats_per_client"]
    sample_weights = core["sample_weights"]
    epoch_weights = core["epoch_weights"]
    val_weights = core["val_weights"]
    outputs = {
        "simple_avg": evaluate_merged_model(simple_average(client_models), client_results, train_dataset, test_loader, config),
        "sample_weighted_avg": evaluate_merged_model(weighted_average(client_models, sample_weights), client_results, train_dataset, test_loader, config),
        "epoch_weighted_avg": evaluate_merged_model(weighted_average(client_models, epoch_weights), client_results, train_dataset, test_loader, config),
        "val_loss_weighted_avg": evaluate_merged_model(weighted_average(client_models, val_weights), client_results, train_dataset, test_loader, config),
        "regmean_equal": evaluate_merged_model(regmean_merge(client_models, stats_per_client, [1.0] * len(client_models), config.regmean_lambda), client_results, train_dataset, test_loader, config),
        "regmean_sample_weighted": evaluate_merged_model(regmean_merge(client_models, stats_per_client, sample_weights, config.regmean_lambda), client_results, train_dataset, test_loader, config),
        "regmean_epoch_weighted": evaluate_merged_model(regmean_merge(client_models, stats_per_client, epoch_weights, config.regmean_lambda), client_results, train_dataset, test_loader, config),
        "regmean_val_loss_weighted": evaluate_merged_model(regmean_merge(client_models, stats_per_client, val_weights, config.regmean_lambda), client_results, train_dataset, test_loader, config),
    }
    return {"config": config_to_jsonable(config), "merges": outputs}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows: List[Dict[str, object]] = []

    group_runners = {
        "few_shot_recovery": few_shot_recovery_results,
        "extreme_non_iid": extreme_non_iid_results,
        "gap_decomposition": gap_decomposition_results,
        "layerwise_ablation": layerwise_ablation_results,
        "weighting_ablation": weighting_ablation_results,
    }

    for seed in args.seeds:
        base_config = make_base_config(args, seed)
        for group in args.groups:
            print(f"[mnist-adv] seed={seed} group={group}", flush=True)
            result = group_runners[group](base_config)
            save_json(args.output_dir / group / f"seed_{seed}.json", result)
            flat_rows.extend(
                [flatten_summary_row(group, group, seed, method, metrics) for method, metrics in result["merges"].items()]
            )

    write_csv(args.output_dir / "summary_rows.csv", flat_rows)
    aggregate = aggregate_rows(flat_rows)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate)
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    build_report_markdown(args.output_dir / "report.md", aggregate, GROUP_NOTES)
    print(json.dumps({"output_dir": str(args.output_dir), "groups": args.groups}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
