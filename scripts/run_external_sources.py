"""
Run NAC on external sources (DeepG or OODRobustBench) using official APIs.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch

from src.external_sources import DeepGAttackGenerator, OODRobustBenchSource
from src.loader import get_resnet18
from src.nac_efficient import EfficientNACAnalyzer


def run_deepg(args):
    generator = DeepGAttackGenerator(
        config_path=args.deepg_config,
        deepg_root=args.deepg_root,
        attack_index=args.attack_index,
    )

    if not generator.is_available():
        raise FileNotFoundError(
            "DeepG libgeometric.so not found. Build it with: (cd libs/deepg && make shared_object)"
        )

    indices = list(range(args.n_examples))
    loader = generator.as_dataloader(indices, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_resnet18().to(device)
    model.eval()

    target_layer = args.layer_name
    analyzer = EfficientNACAnalyzer(model, [target_layer], device)
    analyzer.profile(loader, max_samples=min(args.profile_samples, args.n_examples))

    all_scores = []
    for x, _ in loader:
        x = x.to(device)
        scores = analyzer.score_batch(x)
        all_scores.extend(scores[target_layer].cpu().numpy().tolist())

    print(f"DeepG: n={len(all_scores)} NAC mean={sum(all_scores)/len(all_scores):.6f}")


def run_oodrb(args):
    source = OODRobustBenchSource(data_dir=args.data_dir)

    if args.corruption is not None:
        x, y = source.load_corruption(
            dataset=args.dataset,
            corruption=args.corruption,
            severity=args.severity,
            n_examples=args.n_examples,
            model_name=args.model_name,
            threat_model=args.threat_model,
        )
    elif args.natural_shift is not None:
        x, y = source.load_natural_shift(
            dataset=args.dataset,
            shift=args.natural_shift,
            n_examples=args.n_examples,
            model_name=args.model_name,
            threat_model=args.threat_model,
        )
    else:
        raise ValueError("Specify either --corruption or --natural-shift for OODRobustBench source.")

    loader = source.as_dataloader(x, y, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_resnet18().to(device)
    model.eval()

    target_layer = args.layer_name
    analyzer = EfficientNACAnalyzer(model, [target_layer], device)
    analyzer.profile(loader, max_samples=min(args.profile_samples, args.n_examples))

    all_scores = []
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        scores = analyzer.score_batch(x_batch)
        all_scores.extend(scores[target_layer].cpu().numpy().tolist())

    print(f"OODRobustBench: n={len(all_scores)} NAC mean={sum(all_scores)/len(all_scores):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["deepg", "oodrb"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--profile-samples", type=int, default=256)
    parser.add_argument("--n-examples", type=int, default=256)
    parser.add_argument("--layer-name", type=str, default="block3")

    parser.add_argument("--deepg-config", type=str, default="configs/deepg_cifar10_rotate_small.txt")
    parser.add_argument("--deepg-root", type=str, default=None)
    parser.add_argument("--attack-index", type=int, default=0)

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="Standard")
    parser.add_argument("--threat-model", type=str, default="Linf")
    parser.add_argument("--corruption", type=str, default=None)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--natural-shift", type=str, default=None)

    args = parser.parse_args()

    if args.source == "deepg":
        run_deepg(args)
    else:
        run_oodrb(args)


if __name__ == "__main__":
    main()
