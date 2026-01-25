import importlib.util
import sys
import os

def load_official_nac():
    # 将 root 加入 sys.path 为相对导入提供上下文
    root = os.path.join(os.getcwd(), 'ood_coverage')
    if root not in sys.path:
        sys.path.append(root)
    
    # 路径
    module_path = os.path.join(root, 'openood/postprocessors/nac_postprocessor.py')
    module_name = 'openood.postprocessors.nac_postprocessor'
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    # 手动注入到 sys.modules 以前满足后续可能的内部引用
    sys.modules[module_name] = module
    
    # 执行模块代码
    spec.loader.exec_module(module)
    return module.NACPostprocessor

try:
    NACPostprocessor = load_official_nac()
    print("SUCCESS: Loaded NACPostprocessor via importlib")
except Exception as e:
    import traceback
    traceback.print_exc()
