#!/usr/bin/env python3
"""
Gap diagnosis experiments for MNIST + MLP merge.
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
    class_center_cosine,
    collect_hidden_state_batches,
    collect_logits,
    collect_penultimate_features,
    collect_regmean_stats,
    compute_client_validation_losses,
    config_to_jsonable,
    evaluate_merged_model,
    fit_bias_only_calibration,
    fit_ridge_probe,
    fit_temperature_calibration,
    fit_vector_scaling_calibration,
    linear_cka,
    make_calibration_loader,
    make_class_balanced_subset,
    mean_feature_cosine,
    post_merge_finetune,
    prediction_agreement_from_logits,
    regmean_merge,
    regmean_merge_selected_layers,
    save_json,
    set_seed,
    simple_average,
    symmetric_kl_from_logits,
    train_centralized,
    train_clients,
    validation_loss_weights,
    weighted_average,
)
from report_utils_v2 import aggregate_rows, build_report_markdown, flatten_summary_row, write_csv


GROUP_NOTES = {
    "gap_sources": "Disentangle merge gap, head mismatch, and representation mismatch under a fixed non-IID setting.",
    "function_space": "Compare merged models to centralized in logit space and prediction space.",
    "representation_space": "Compare hidden-layer and penultimate-feature alignment to centralized.",
    "gap_reduction": "Test concrete modifications to shrink the gap to centralized.",
    "systematic_calibration": "Systematic post-merge calibration around the sample-weighted and regmean-hidden-layers mainline.",
    "calibration_strength": "Progressively strengthen post-merge calibration to test whether the remaining gap is more than confidence calibration.",
    "layerwise_alignment": "Measure where representation mismatch starts: early hidden states, deeper hidden states, or penultimate features.",
    "layerwise_intervention": "Restrict tuning or selective RegMean to specific layers and test which interventions shrink the gap most.",
    "function_repr_bridge": "Track function-space and representation-space changes together before and after light post-merge interventions.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gap diagnosis experiments")
    parser.add_argument("--data-root", type=Path, default=Path("./work/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/gap_diagnostics_suite"))
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
            "gap_sources",
            "function_space",
            "representation_space",
            "gap_reduction",
            "systematic_calibration",
            "calibration_strength",
            "layerwise_alignment",
            "layerwise_intervention",
            "function_repr_bridge",
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
        split_mode="dirichlet",
        dirichlet_alpha=0.5,
        client_epochs=tuple([1, 2, 3, 5, 8, 10, 15, 20, 30, 50]),
        centralized_epochs=10,
    )


def prepare_models(config: ExperimentConfig):
    set_seed(config.seed)
    train_dataset, test_dataset = build_datasets(config.data_root)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)
    base_model = SmallMLP(hidden_sizes=config.hidden_sizes)
    client_results = train_clients(config, train_dataset, base_model)
    client_models = [client.model for client in client_results]
    sample_weights = [len(client.spec.indices) for client in client_results]
    epoch_weights = [client.spec.epochs for client in client_results]
    val_weights = validation_loss_weights(compute_client_validation_losses(client_results, train_dataset, config))
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
    models = {
        "simple_avg": simple_average(client_models),
        "sample_weighted_avg": weighted_average(client_models, sample_weights),
        "regmean_equal": regmean_merge(client_models, stats_per_client, [1.0] * len(client_models), config.regmean_lambda),
        "regmean_sample_weighted": regmean_merge(client_models, stats_per_client, sample_weights, config.regmean_lambda),
        "regmean_hidden_layers": regmean_merge_selected_layers(client_models, stats_per_client, sample_weights, config.regmean_lambda, [0, 1]),
        "centralized": train_centralized(config, train_dataset, base_model),
    }
    return {
        "config": config,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "test_loader": test_loader,
        "client_results": client_results,
        "models": models,
        "sample_weights": sample_weights,
        "epoch_weights": epoch_weights,
        "val_weights": val_weights,
    }


def build_standard_rows(group: str, seed: int, payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for method, metrics in payload["merges"].items():
        rows.append(flatten_summary_row(group, group, seed, method, metrics))
    return rows


def build_rows_from_settings(group: str, seed: int, payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for setting, setting_payload in payload["settings"].items():
        for method, metrics in setting_payload["merges"].items():
            rows.append(flatten_summary_row(group, setting, seed, method, metrics))
    return rows


def build_analysis_models(core: Dict[str, object], config: ExperimentConfig, head_per_class: int = 20, hidden_per_class: int = 50) -> Dict[str, SmallMLP]:
    train_dataset = core["train_dataset"]
    models = dict(core["models"])
    head_tune_indices = make_class_balanced_subset(train_dataset, head_per_class, config.seed + 1300 + head_per_class)
    hidden_tune_indices = make_class_balanced_subset(train_dataset, hidden_per_class, config.seed + 2300 + hidden_per_class)
    models["regmean_hidden_layers_last_layer_tune"] = post_merge_finetune(
        models["regmean_hidden_layers"],
        train_dataset,
        head_tune_indices,
        config,
        steps=100,
        lr=1e-4,
        layers_to_train=[2],
    )
    models["regmean_hidden_layers_hidden_head_tune"] = post_merge_finetune(
        models["regmean_hidden_layers"],
        train_dataset,
        hidden_tune_indices,
        config,
        steps=120,
        lr=8e-5,
        layers_to_train=[1, 2],
    )
    models["sample_weighted_avg_last_layer_tune"] = post_merge_finetune(
        models["sample_weighted_avg"],
        train_dataset,
        head_tune_indices,
        config,
        steps=100,
        lr=1e-4,
        layers_to_train=[2],
    )
    models["sample_weighted_avg_hidden_head_tune"] = post_merge_finetune(
        models["sample_weighted_avg"],
        train_dataset,
        hidden_tune_indices,
        config,
        steps=120,
        lr=8e-5,
        layers_to_train=[1, 2],
    )
    return models


def function_repr_snapshot(
    model: SmallMLP,
    train_dataset,
    eval_indices: List[int],
    ref_logits: torch.Tensor,
    ref_hidden: List[torch.Tensor],
    ref_penultimate: torch.Tensor,
    labels: torch.Tensor,
    config: ExperimentConfig,
    is_reference: bool = False,
) -> Dict[str, float]:
    logits, eval_labels = collect_logits(model, train_dataset, eval_indices, config)
    hidden = collect_hidden_state_batches(model, train_dataset, eval_indices, config)
    penultimate, _ = collect_penultimate_features(model, train_dataset, eval_indices, config)
    layer0_cos = 1.0 if is_reference else mean_feature_cosine(hidden[0], ref_hidden[0])
    penultimate_cos = 1.0 if is_reference else mean_feature_cosine(penultimate, ref_penultimate)
    return {
        "subset_accuracy": float((logits.argmax(dim=1) == eval_labels).float().mean().item()),
        "symmetric_kl_to_centralized": 0.0 if is_reference else symmetric_kl_from_logits(logits, ref_logits),
        "prediction_agreement_to_centralized": prediction_agreement_from_logits(logits, ref_logits),
        "layer0_cosine_to_centralized": layer0_cos,
        "layer1_cosine_to_centralized": 1.0 if is_reference else mean_feature_cosine(hidden[1], ref_hidden[1]),
        "layer2_cosine_to_centralized": 1.0 if is_reference else mean_feature_cosine(hidden[2], ref_hidden[2]),
        "penultimate_cosine_to_centralized": penultimate_cos,
        "layer0_cka_to_centralized": 1.0 if is_reference else linear_cka(hidden[0], ref_hidden[0]),
        "layer1_cka_to_centralized": 1.0 if is_reference else linear_cka(hidden[1], ref_hidden[1]),
        "layer2_cka_to_centralized": 1.0 if is_reference else linear_cka(hidden[2], ref_hidden[2]),
        "penultimate_cka_to_centralized": 1.0 if is_reference else linear_cka(penultimate, ref_penultimate),
        "layer0_center_cosine_to_centralized": 1.0 if is_reference else class_center_cosine(hidden[0], ref_hidden[0], labels),
        "layer1_center_cosine_to_centralized": 1.0 if is_reference else class_center_cosine(hidden[1], ref_hidden[1], labels),
        "layer2_center_cosine_to_centralized": 1.0 if is_reference else class_center_cosine(hidden[2], ref_hidden[2], labels),
        "penultimate_center_cosine_to_centralized": 1.0 if is_reference else class_center_cosine(penultimate, ref_penultimate, labels),
        "deep_representation_drop": layer0_cos - penultimate_cos,
    }


def gap_sources_result(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_models(config)
    train_dataset = core["train_dataset"]
    test_loader = core["test_loader"]
    client_results = core["client_results"]
    models = core["models"]
    calibration_indices = make_class_balanced_subset(train_dataset, 10, config.seed + 101)
    more_indices = make_class_balanced_subset(train_dataset, 50, config.seed + 202)

    merges: Dict[str, object] = {}
    for name in ["simple_avg", "regmean_equal", "sample_weighted_avg", "centralized"]:
        model = models[name]
        merges[name] = evaluate_merged_model(model, client_results, train_dataset, test_loader, config)
        if name == "centralized":
            continue
        merges[f"{name}_bias_cal"] = evaluate_merged_model(
            fit_bias_only_calibration(model, train_dataset, calibration_indices, config, steps=40, lr=3e-3),
            client_results,
            train_dataset,
            test_loader,
            config,
        )
        merges[f"{name}_ridge_head"] = evaluate_merged_model(
            fit_ridge_probe(model, train_dataset, calibration_indices, config, ridge_lambda=5.0),
            client_results,
            train_dataset,
            test_loader,
            config,
        )
        merges[f"{name}_short_tune"] = evaluate_merged_model(
            post_merge_finetune(model, train_dataset, more_indices, config, steps=80, lr=1e-4, layers_to_train=[2]),
            client_results,
            train_dataset,
            test_loader,
            config,
        )
    return {"config": config_to_jsonable(config), "merges": merges}


def function_space_result(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_models(config)
    train_dataset = core["train_dataset"]
    models = build_analysis_models(core, config)
    eval_indices = make_class_balanced_subset(train_dataset, 100, config.seed + 303)
    ref_logits, labels = collect_logits(models["centralized"], train_dataset, eval_indices, config)
    diagnostics: Dict[str, object] = {}
    for name in [
        "simple_avg",
        "sample_weighted_avg",
        "regmean_equal",
        "regmean_sample_weighted",
        "regmean_hidden_layers",
        "regmean_hidden_layers_last_layer_tune",
        "regmean_hidden_layers_hidden_head_tune",
        "centralized",
    ]:
        logits, _ = collect_logits(models[name], train_dataset, eval_indices, config)
        diagnostics[name] = {
            "symmetric_kl_to_centralized": 0.0 if name == "centralized" else symmetric_kl_from_logits(logits, ref_logits),
            "prediction_agreement_to_centralized": prediction_agreement_from_logits(logits, ref_logits),
            "self_accuracy_on_eval_subset": float((logits.argmax(dim=1) == labels).float().mean().item()),
        }
    return {"config": config_to_jsonable(config), "diagnostics": diagnostics}


def representation_space_result(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_models(config)
    train_dataset = core["train_dataset"]
    models = build_analysis_models(core, config)
    eval_indices = make_class_balanced_subset(train_dataset, 100, config.seed + 404)
    ref_hidden = collect_hidden_state_batches(models["centralized"], train_dataset, eval_indices, config)
    ref_penultimate, _ = collect_penultimate_features(models["centralized"], train_dataset, eval_indices, config)
    diagnostics: Dict[str, object] = {}
    for name in [
        "simple_avg",
        "sample_weighted_avg",
        "regmean_equal",
        "regmean_sample_weighted",
        "regmean_hidden_layers",
        "regmean_hidden_layers_last_layer_tune",
        "regmean_hidden_layers_hidden_head_tune",
        "centralized",
    ]:
        hidden = collect_hidden_state_batches(models[name], train_dataset, eval_indices, config)
        penultimate, _ = collect_penultimate_features(models[name], train_dataset, eval_indices, config)
        layer0 = 1.0 if name == "centralized" else mean_feature_cosine(hidden[0], ref_hidden[0])
        penultimate_cos = 1.0 if name == "centralized" else mean_feature_cosine(penultimate, ref_penultimate)
        diagnostics[name] = {
            "penultimate_cosine_to_centralized": penultimate_cos,
            "layer0_cosine_to_centralized": layer0,
            "layer1_cosine_to_centralized": 1.0 if name == "centralized" else mean_feature_cosine(hidden[1], ref_hidden[1]),
            "layer2_cosine_to_centralized": 1.0 if name == "centralized" else mean_feature_cosine(hidden[2], ref_hidden[2]),
            "deep_representation_drop": layer0 - penultimate_cos,
        }
    return {"config": config_to_jsonable(config), "diagnostics": diagnostics}


def gap_reduction_result(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_models(config)
    train_dataset = core["train_dataset"]
    test_loader = core["test_loader"]
    client_results = core["client_results"]
    models = core["models"]
    calibration_indices = make_class_balanced_subset(train_dataset, 20, config.seed + 505)
    more_indices = make_class_balanced_subset(train_dataset, 50, config.seed + 606)
    merges: Dict[str, object] = {
        "simple_avg": evaluate_merged_model(models["simple_avg"], client_results, train_dataset, test_loader, config),
        "sample_weighted_avg": evaluate_merged_model(models["sample_weighted_avg"], client_results, train_dataset, test_loader, config),
        "regmean_equal": evaluate_merged_model(models["regmean_equal"], client_results, train_dataset, test_loader, config),
        "regmean_sample_weighted": evaluate_merged_model(models["regmean_sample_weighted"], client_results, train_dataset, test_loader, config),
        "regmean_hidden_layers": evaluate_merged_model(models["regmean_hidden_layers"], client_results, train_dataset, test_loader, config),
        "centralized": evaluate_merged_model(models["centralized"], client_results, train_dataset, test_loader, config),
    }
    candidate_models = {
        "simple_avg_ridge_head": fit_ridge_probe(models["simple_avg"], train_dataset, calibration_indices, config, ridge_lambda=5.0),
        "regmean_sample_weighted_ridge_head": fit_ridge_probe(models["regmean_sample_weighted"], train_dataset, calibration_indices, config, ridge_lambda=5.0),
        "regmean_hidden_layers_bias_cal": fit_bias_only_calibration(models["regmean_hidden_layers"], train_dataset, calibration_indices, config, steps=40, lr=3e-3),
        "regmean_hidden_layers_short_tune": post_merge_finetune(models["regmean_hidden_layers"], train_dataset, more_indices, config, steps=80, lr=1e-4, layers_to_train=[2]),
    }
    for name, model in candidate_models.items():
        merges[name] = evaluate_merged_model(model, client_results, train_dataset, test_loader, config)
    return {"config": config_to_jsonable(config), "merges": merges}


def systematic_calibration_result(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_models(config)
    train_dataset = core["train_dataset"]
    test_loader = core["test_loader"]
    client_results = core["client_results"]
    models = core["models"]
    settings: Dict[str, object] = {}
    budgets = [5, 10, 20, 50]

    for per_class in budgets:
        calibration_indices = make_class_balanced_subset(train_dataset, per_class, config.seed + 3000 + per_class)
        hidden_tune_indices = make_class_balanced_subset(train_dataset, max(per_class, 20), config.seed + 4000 + per_class)
        setting_name = f"{per_class}shot_per_class"
        candidate_models = {
            "sample_weighted_avg": models["sample_weighted_avg"],
            "sample_weighted_bias_cal": fit_bias_only_calibration(models["sample_weighted_avg"], train_dataset, calibration_indices, config, steps=50, lr=3e-3),
            "sample_weighted_temperature": fit_temperature_calibration(models["sample_weighted_avg"], train_dataset, calibration_indices, config, steps=80, lr=1e-2),
            "sample_weighted_vector_scaling": fit_vector_scaling_calibration(models["sample_weighted_avg"], train_dataset, calibration_indices, config, steps=100, lr=8e-3),
            "sample_weighted_last_layer_tune": post_merge_finetune(models["sample_weighted_avg"], train_dataset, calibration_indices, config, steps=100, lr=1e-4, layers_to_train=[2]),
            "sample_weighted_hidden_head_tune": post_merge_finetune(models["sample_weighted_avg"], train_dataset, hidden_tune_indices, config, steps=140, lr=8e-5, layers_to_train=[1, 2]),
            "regmean_hidden_layers": models["regmean_hidden_layers"],
            "regmean_hidden_layers_bias_cal": fit_bias_only_calibration(models["regmean_hidden_layers"], train_dataset, calibration_indices, config, steps=50, lr=3e-3),
            "regmean_hidden_layers_temperature": fit_temperature_calibration(models["regmean_hidden_layers"], train_dataset, calibration_indices, config, steps=80, lr=1e-2),
            "regmean_hidden_layers_vector_scaling": fit_vector_scaling_calibration(models["regmean_hidden_layers"], train_dataset, calibration_indices, config, steps=100, lr=8e-3),
            "regmean_hidden_layers_last_layer_tune": post_merge_finetune(models["regmean_hidden_layers"], train_dataset, calibration_indices, config, steps=100, lr=1e-4, layers_to_train=[2]),
            "regmean_hidden_layers_hidden_head_tune": post_merge_finetune(models["regmean_hidden_layers"], train_dataset, hidden_tune_indices, config, steps=140, lr=8e-5, layers_to_train=[1, 2]),
            "centralized": models["centralized"],
        }
        merges = {
            name: evaluate_merged_model(model, client_results, train_dataset, test_loader, config)
            for name, model in candidate_models.items()
        }
        settings[setting_name] = {"calibration_per_class": per_class, "merges": merges}

    return {"config": config_to_jsonable(config), "settings": settings}


def calibration_strength_result(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_models(config)
    train_dataset = core["train_dataset"]
    test_loader = core["test_loader"]
    client_results = core["client_results"]
    models = core["models"]
    settings: Dict[str, object] = {}

    for per_class in [5, 10, 20, 50]:
        calibration_indices = make_class_balanced_subset(train_dataset, per_class, config.seed + 5000 + per_class)
        hidden_indices = make_class_balanced_subset(train_dataset, max(per_class, 20), config.seed + 6000 + per_class)
        setting_name = f"{per_class}shot_per_class"
        candidates = {
            "sample_weighted_avg": models["sample_weighted_avg"],
            "sample_weighted_temperature": fit_temperature_calibration(models["sample_weighted_avg"], train_dataset, calibration_indices, config, steps=80, lr=1e-2),
            "sample_weighted_bias_cal": fit_bias_only_calibration(models["sample_weighted_avg"], train_dataset, calibration_indices, config, steps=50, lr=3e-3),
            "sample_weighted_vector_scaling": fit_vector_scaling_calibration(models["sample_weighted_avg"], train_dataset, calibration_indices, config, steps=100, lr=8e-3),
            "sample_weighted_last_layer_tune": post_merge_finetune(models["sample_weighted_avg"], train_dataset, calibration_indices, config, steps=100, lr=1e-4, layers_to_train=[2]),
            "sample_weighted_hidden_head_tune": post_merge_finetune(models["sample_weighted_avg"], train_dataset, hidden_indices, config, steps=140, lr=8e-5, layers_to_train=[1, 2]),
            "regmean_hidden_layers": models["regmean_hidden_layers"],
            "regmean_hidden_layers_temperature": fit_temperature_calibration(models["regmean_hidden_layers"], train_dataset, calibration_indices, config, steps=80, lr=1e-2),
            "regmean_hidden_layers_bias_cal": fit_bias_only_calibration(models["regmean_hidden_layers"], train_dataset, calibration_indices, config, steps=50, lr=3e-3),
            "regmean_hidden_layers_vector_scaling": fit_vector_scaling_calibration(models["regmean_hidden_layers"], train_dataset, calibration_indices, config, steps=100, lr=8e-3),
            "regmean_hidden_layers_last_layer_tune": post_merge_finetune(models["regmean_hidden_layers"], train_dataset, calibration_indices, config, steps=100, lr=1e-4, layers_to_train=[2]),
            "regmean_hidden_layers_hidden_head_tune": post_merge_finetune(models["regmean_hidden_layers"], train_dataset, hidden_indices, config, steps=140, lr=8e-5, layers_to_train=[1, 2]),
            "centralized": models["centralized"],
        }
        merges = {
            name: evaluate_merged_model(model, client_results, train_dataset, test_loader, config)
            for name, model in candidates.items()
        }
        settings[setting_name] = {"calibration_per_class": per_class, "merges": merges}
    return {"config": config_to_jsonable(config), "settings": settings}


def layerwise_alignment_result(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_models(config)
    train_dataset = core["train_dataset"]
    models = build_analysis_models(core, config)
    eval_indices = make_class_balanced_subset(train_dataset, 100, config.seed + 7001)
    ref_logits, labels = collect_logits(models["centralized"], train_dataset, eval_indices, config)
    ref_hidden = collect_hidden_state_batches(models["centralized"], train_dataset, eval_indices, config)
    ref_penultimate, _ = collect_penultimate_features(models["centralized"], train_dataset, eval_indices, config)
    diagnostics: Dict[str, object] = {}
    for name in [
        "simple_avg",
        "sample_weighted_avg",
        "sample_weighted_avg_last_layer_tune",
        "sample_weighted_avg_hidden_head_tune",
        "regmean_equal",
        "regmean_sample_weighted",
        "regmean_hidden_layers",
        "regmean_hidden_layers_last_layer_tune",
        "regmean_hidden_layers_hidden_head_tune",
        "centralized",
    ]:
        diagnostics[name] = function_repr_snapshot(
            models[name],
            train_dataset,
            eval_indices,
            ref_logits,
            ref_hidden,
            ref_penultimate,
            labels,
            config,
            is_reference=name == "centralized",
        )
    return {"config": config_to_jsonable(config), "diagnostics": diagnostics}


def layerwise_intervention_result(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_models(config)
    train_dataset = core["train_dataset"]
    test_loader = core["test_loader"]
    client_results = core["client_results"]
    models = core["models"]
    client_models = [client.model for client in client_results]
    sample_weights = core["sample_weights"]
    stats_per_client = [
        collect_regmean_stats(
            client.model,
            make_calibration_loader(
                train_dataset,
                client.spec.indices,
                config.calibration_size,
                config.eval_batch_size,
                config.seed + 8000 + client.spec.client_id,
            ),
            torch.device(config.device),
        )
        for client in client_results
    ]
    tune_indices = make_class_balanced_subset(train_dataset, 20, config.seed + 8002)
    hidden_tune_indices = make_class_balanced_subset(train_dataset, 50, config.seed + 8003)
    settings: Dict[str, object] = {
        "merge_layer_selection": {
            "merges": {
                "simple_avg": evaluate_merged_model(models["simple_avg"], client_results, train_dataset, test_loader, config),
                "sample_weighted_avg": evaluate_merged_model(models["sample_weighted_avg"], client_results, train_dataset, test_loader, config),
                "regmean_hidden_layers": evaluate_merged_model(regmean_merge_selected_layers(client_models, stats_per_client, sample_weights, config.regmean_lambda, [0, 1]), client_results, train_dataset, test_loader, config),
                "regmean_last_layer": evaluate_merged_model(regmean_merge_selected_layers(client_models, stats_per_client, sample_weights, config.regmean_lambda, [2]), client_results, train_dataset, test_loader, config),
                "regmean_middle_last": evaluate_merged_model(regmean_merge_selected_layers(client_models, stats_per_client, sample_weights, config.regmean_lambda, [1, 2]), client_results, train_dataset, test_loader, config),
                "regmean_all_layers": evaluate_merged_model(regmean_merge(client_models, stats_per_client, sample_weights, config.regmean_lambda), client_results, train_dataset, test_loader, config),
                "centralized": evaluate_merged_model(models["centralized"], client_results, train_dataset, test_loader, config),
            }
        },
        "tune_layer_selection": {
            "merges": {
                "sample_weighted_avg": evaluate_merged_model(models["sample_weighted_avg"], client_results, train_dataset, test_loader, config),
                "sample_weighted_last_layer_tune": evaluate_merged_model(post_merge_finetune(models["sample_weighted_avg"], train_dataset, tune_indices, config, steps=100, lr=1e-4, layers_to_train=[2]), client_results, train_dataset, test_loader, config),
                "sample_weighted_middle_layer_tune": evaluate_merged_model(post_merge_finetune(models["sample_weighted_avg"], train_dataset, hidden_tune_indices, config, steps=120, lr=8e-5, layers_to_train=[1]), client_results, train_dataset, test_loader, config),
                "sample_weighted_hidden_head_tune": evaluate_merged_model(post_merge_finetune(models["sample_weighted_avg"], train_dataset, hidden_tune_indices, config, steps=140, lr=8e-5, layers_to_train=[1, 2]), client_results, train_dataset, test_loader, config),
                "sample_weighted_full_tune": evaluate_merged_model(post_merge_finetune(models["sample_weighted_avg"], train_dataset, hidden_tune_indices, config, steps=140, lr=8e-5, layers_to_train=[0, 1, 2]), client_results, train_dataset, test_loader, config),
                "regmean_hidden_layers": evaluate_merged_model(models["regmean_hidden_layers"], client_results, train_dataset, test_loader, config),
                "regmean_hidden_last_layer_tune": evaluate_merged_model(post_merge_finetune(models["regmean_hidden_layers"], train_dataset, tune_indices, config, steps=100, lr=1e-4, layers_to_train=[2]), client_results, train_dataset, test_loader, config),
                "regmean_hidden_middle_layer_tune": evaluate_merged_model(post_merge_finetune(models["regmean_hidden_layers"], train_dataset, hidden_tune_indices, config, steps=120, lr=8e-5, layers_to_train=[1]), client_results, train_dataset, test_loader, config),
                "regmean_hidden_hidden_head_tune": evaluate_merged_model(post_merge_finetune(models["regmean_hidden_layers"], train_dataset, hidden_tune_indices, config, steps=140, lr=8e-5, layers_to_train=[1, 2]), client_results, train_dataset, test_loader, config),
                "regmean_hidden_full_tune": evaluate_merged_model(post_merge_finetune(models["regmean_hidden_layers"], train_dataset, hidden_tune_indices, config, steps=140, lr=8e-5, layers_to_train=[0, 1, 2]), client_results, train_dataset, test_loader, config),
                "centralized": evaluate_merged_model(models["centralized"], client_results, train_dataset, test_loader, config),
            }
        },
    }
    return {"config": config_to_jsonable(config), "settings": settings}


def function_repr_bridge_result(config: ExperimentConfig) -> Dict[str, object]:
    core = prepare_models(config)
    train_dataset = core["train_dataset"]
    models = build_analysis_models(core, config)
    eval_indices = make_class_balanced_subset(train_dataset, 100, config.seed + 9001)
    ref_logits, labels = collect_logits(models["centralized"], train_dataset, eval_indices, config)
    ref_hidden = collect_hidden_state_batches(models["centralized"], train_dataset, eval_indices, config)
    ref_penultimate, _ = collect_penultimate_features(models["centralized"], train_dataset, eval_indices, config)
    diagnostics: Dict[str, object] = {}
    for name in [
        "sample_weighted_avg",
        "sample_weighted_avg_last_layer_tune",
        "sample_weighted_avg_hidden_head_tune",
        "regmean_hidden_layers",
        "regmean_hidden_layers_last_layer_tune",
        "regmean_hidden_layers_hidden_head_tune",
        "centralized",
    ]:
        diagnostics[name] = function_repr_snapshot(
            models[name],
            train_dataset,
            eval_indices,
            ref_logits,
            ref_hidden,
            ref_penultimate,
            labels,
            config,
            is_reference=name == "centralized",
        )
    return {"config": config_to_jsonable(config), "diagnostics": diagnostics}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows: List[Dict[str, object]] = []
    diagnostics_payload: Dict[str, object] = {}
    runners = {
        "gap_sources": gap_sources_result,
        "function_space": function_space_result,
        "representation_space": representation_space_result,
        "gap_reduction": gap_reduction_result,
        "systematic_calibration": systematic_calibration_result,
        "calibration_strength": calibration_strength_result,
        "layerwise_alignment": layerwise_alignment_result,
        "layerwise_intervention": layerwise_intervention_result,
        "function_repr_bridge": function_repr_bridge_result,
    }

    for seed in args.seeds:
        config = make_base_config(args, seed)
        for group in args.groups:
            print(f"[gap-diagnostics] seed={seed} group={group}", flush=True)
            result = runners[group](deepcopy(config))
            save_json(args.output_dir / group / f"seed_{seed}.json", result)
            if "merges" in result:
                flat_rows.extend(build_standard_rows(group, seed, result))
            if "settings" in result:
                flat_rows.extend(build_rows_from_settings(group, seed, result))
            if "diagnostics" in result:
                diagnostics_payload.setdefault(group, {})[str(seed)] = result["diagnostics"]

    if flat_rows:
        write_csv(args.output_dir / "summary_rows.csv", flat_rows)
        aggregate = aggregate_rows(flat_rows)
        write_csv(args.output_dir / "aggregate_summary.csv", aggregate)
        (args.output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
        build_report_markdown(args.output_dir / "report.md", aggregate, GROUP_NOTES)
    if diagnostics_payload:
        (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "groups": args.groups,
                "files": [
                    str(args.output_dir / "summary_rows.csv"),
                    str(args.output_dir / "aggregate_summary.csv"),
                    str(args.output_dir / "report.md"),
                    str(args.output_dir / "diagnostics.json"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
