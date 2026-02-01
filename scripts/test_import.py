import sys
import os
from unittest.mock import MagicMock

# Mock optional OpenOOD dependencies.
missing_deps = [
    'faiss', 'diffdist', 'libmr', 'cvxpy', 'cvxopt', 
    'scikit-learn', 'sklearn', 'pandas', 'openpyxl'
]
for dep in missing_deps:
    sys.modules[dep] = MagicMock()

# Add local OpenOOD path.
curr_dir = os.getcwd()
sys.path.append(os.path.join(curr_dir, 'ood_coverage'))

try:
    from openood.postprocessors.nac_postprocessor import NACPostprocessor
    from openood.postprocessors.nac.coverage import Estimator, KMNC
    print("SUCCESS: Imported Official Classes")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILED: {e}")
