"""
Print candidate layer names for a RobustBench model.
Usage:
  python3 scripts/inspect_rb_layers.py --model-name Standard --dataset cifar10 --threat-model Linf
"""
import argparse
from robustbench import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--threat-model", default="Linf")
    args = parser.parse_args()

    model = load_model(
        model_name=args.model_name,
        dataset=args.dataset,
        threat_model=args.threat_model,
    )

    print(f"Model: {args.model_name} ({args.dataset}, {args.threat_model})")
    for name, _ in model.named_modules():
        if name:
            print(name)


if __name__ == "__main__":
    main()
