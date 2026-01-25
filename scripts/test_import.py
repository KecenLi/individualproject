import sys
import os
from unittest.mock import MagicMock

# 模拟 OpenOOD 复杂的依赖
missing_deps = [
    'faiss', 'diffdist', 'libmr', 'cvxpy', 'cvxopt', 
    'scikit-learn', 'sklearn', 'pandas', 'openpyxl'
]
for dep in missing_deps:
    sys.modules[dep] = MagicMock()

# 将路径加入
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
