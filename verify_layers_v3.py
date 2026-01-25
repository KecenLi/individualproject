
import torch
from robustbench import load_model

def print_last_layers(model_name):
    print(f"\nLast layers for {model_name}:")
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    names = [name for name, _ in model.named_modules() if 'layer' in name]
    for n in names[-20:]:
        print(n)

print_last_layers('Standard')
print_last_layers('Gowal2021Improving_28_10_ddpm_100m')
