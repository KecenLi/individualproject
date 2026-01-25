
import torch
from robustbench import load_model

def print_layers(model_name):
    print(f"\nLayers for {model_name}:")
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    # Use nethook style names through ood_coverage if possible, or just print named_modules
    for name, _ in model.named_modules():
        if name in ['layer4.1', 'layer3.3', 'layer4', 'layer3']:
            print(f"Found: {name}")

print_layers('Standard')
print_layers('Gowal2021Improving_28_10_ddpm_100m')
