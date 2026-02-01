import importlib.util
import sys
import os

def load_official_nac():
    # Add root to sys.path for relative imports.
    root = os.path.join(os.getcwd(), 'ood_coverage')
    if root not in sys.path:
        sys.path.append(root)
    
    # Module path.
    module_path = os.path.join(root, 'openood/postprocessors/nac_postprocessor.py')
    module_name = 'openood.postprocessors.nac_postprocessor'
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Register module for downstream imports.
    sys.modules[module_name] = module
    
    # Execute module code.
    spec.loader.exec_module(module)
    return module.NACPostprocessor

try:
    NACPostprocessor = load_official_nac()
    print("SUCCESS: Loaded NACPostprocessor via importlib")
except Exception as e:
    import traceback
    traceback.print_exc()
