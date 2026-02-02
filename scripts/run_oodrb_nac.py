"""
Run NAC evaluation on OODRobustBench corruption shifts (and optional natural shifts)
using official OODRobustBench/RobustBench dataset loaders.
"""
import argparse
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from robustbench.data import CORRUPTIONS_DICT
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from src.external_sources import OODRobustBenchSource
from src.loader import get_cifar10_loader, get_imagenet_loader, get_rb_model
from src.official_nac import OfficialNACWrapper
from openood.evaluators.metrics import compute_all_metrics


def _batch_scores(analyzer: OfficialNACWrapper, loader: DataLoader, device: str) -> np.ndarray:
    scores = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        with torch.set_grad_enabled(True):
            s = analyzer.score_batch(x).detach().cpu().numpy()
        scores.append(s)
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="Standard")
    parser.add_argument("--threat-model", type=str, default="Linf")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--profile-samples", type=int, default=1000)
    parser.add_argument("--id-examples", type=int, default=10000)
    parser.add_argument("--layer-names", type=str, default="block3",
                        help="Comma-separated layer names for NAC (e.g., block3 or model.layer3)")
    parser.add_argument("--corruptions", type=str, default="all",
                        help="Comma-separated corruption names or 'all'")
    parser.add_argument("--severities", type=str, default="1,2,3,4,5")
    parser.add_argument("--run-natural-shifts", action="store_true")
    parser.add_argument("--natural-shifts", type=str, default="all")
    parser.add_argument("--output-dir", type=str, default="oodrb_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.output_dir, f"oodrb_nac_{timestamp}.csv")

    # Load model and NAC wrapper
    model = get_rb_model(
        model_name=args.model_name,
        dataset=args.dataset,
        threat_model=args.threat_model,
    ).to(device)
    model.eval()

    # Profiling on CIFAR-10 train split
    if args.dataset == "cifar10":
        train_loader = get_cifar10_loader(
            batch_size=args.batch_size,
            train=True,
            n_examples=args.profile_samples,
            data_dir=args.data_dir,
        )
    elif args.dataset == "imagenet":
        train_loader = get_imagenet_loader(
            batch_size=args.batch_size,
            train=True,
            n_examples=args.profile_samples,
            data_dir=args.data_dir,
        )
    else:
        raise ValueError(f"Unsupported dataset for profiling: {args.dataset}")
    analyzer = OfficialNACWrapper(model, device=device)
    layer_names = [l.strip() for l in args.layer_names.split(",") if l.strip()]
    analyzer.setup(train_loader, layer_names=layer_names, valid_num=args.profile_samples)

    # ID scores for metric baseline
    if args.dataset == "cifar10":
        id_loader = get_cifar10_loader(
            batch_size=args.batch_size,
            train=False,
            n_examples=args.id_examples,
            data_dir=args.data_dir,
        )
    elif args.dataset == "imagenet":
        id_loader = get_imagenet_loader(
            batch_size=args.batch_size,
            train=False,
            n_examples=args.id_examples,
            data_dir=args.data_dir,
        )
    else:
        raise ValueError(f"Unsupported dataset for ID evaluation: {args.dataset}")
    id_scores = _batch_scores(analyzer, id_loader, device)

    source = OODRobustBenchSource(data_dir=args.data_dir)

    # Determine corruption list
    dataset_enum = BenchmarkDataset(args.dataset)
    corruption_list = CORRUPTIONS_DICT[dataset_enum][ThreatModel.corruptions]
    if args.corruptions != "all":
        requested = [c.strip() for c in args.corruptions.split(",") if c.strip()]
        corruption_list = [c for c in corruption_list if c in requested]

    severities = [int(s.strip()) for s in args.severities.split(",") if s.strip()]

    results = []

    # Corruption shifts
    for corruption in corruption_list:
        for severity in severities:
            print(f"[OODRB] corruption={corruption} severity={severity}")
            x, y = source.load_corruption(
                dataset=args.dataset,
                corruption=corruption,
                severity=severity,
                n_examples=args.id_examples,
                model_name=args.model_name,
                threat_model=args.threat_model,
            )
            loader = source.as_dataloader(x, y, batch_size=args.batch_size)
            ood_scores = _batch_scores(analyzer, loader, device)

            combined_scores = np.concatenate([id_scores, ood_scores])
            combined_labels = np.concatenate([np.ones_like(id_scores), -1 * np.ones_like(ood_scores)])
            combined_preds = np.concatenate([np.zeros_like(id_scores), np.zeros_like(ood_scores)])
            metrics = compute_all_metrics(combined_scores, combined_labels, combined_preds)

            results.append({
                "shift_type": "corruption",
                "shift_name": corruption,
                "severity": severity,
                "auroc": float(metrics[1]),
                "fpr95": float(metrics[0]),
                "aupr_in": float(metrics[2]),
                "aupr_out": float(metrics[3]),
            })

    # Natural shifts (optional)
    if args.run_natural_shifts:
        if args.natural_shifts == "all":
            shift_list = source._oodrb_data.NATURAL_SHIFTS[args.dataset]
        else:
            shift_list = [s.strip() for s in args.natural_shifts.split(",") if s.strip()]

        for shift in shift_list:
            print(f"[OODRB] natural_shift={shift}")
            x, y = source.load_natural_shift(
                dataset=args.dataset,
                shift=shift,
                n_examples=args.id_examples,
                model_name=args.model_name,
                threat_model=args.threat_model,
            )
            loader = source.as_dataloader(x, y, batch_size=args.batch_size)
            ood_scores = _batch_scores(analyzer, loader, device)

            combined_scores = np.concatenate([id_scores, ood_scores])
            combined_labels = np.concatenate([np.ones_like(id_scores), -1 * np.ones_like(ood_scores)])
            combined_preds = np.concatenate([np.zeros_like(id_scores), np.zeros_like(ood_scores)])
            metrics = compute_all_metrics(combined_scores, combined_labels, combined_preds)

            results.append({
                "shift_type": "natural",
                "shift_name": shift,
                "severity": None,
                "auroc": float(metrics[1]),
                "fpr95": float(metrics[0]),
                "aupr_in": float(metrics[2]),
                "aupr_out": float(metrics[3]),
            })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Saved: {out_csv}")


if __name__ == "__main__":
    main()
