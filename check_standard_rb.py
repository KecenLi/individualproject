
from robustbench import load_model
try:
    model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
    print("Successfully loaded Standard model from RobustBench.")
except Exception as e:
    print(f"Error loading Standard model: {e}")
