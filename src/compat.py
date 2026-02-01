# Compatibility shim for older torchvision imports.
import torch
import torchvision.models
import types

# Restore torchvision.models.utils for older dependencies.
utils_module = types.ModuleType('torchvision.models.utils')
utils_module.load_state_dict_from_url = torch.hub.load_state_dict_from_url
torchvision.models.utils = utils_module

# Inject into sys.modules for import compatibility.
import sys
sys.modules['torchvision.models.utils'] = utils_module
