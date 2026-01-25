# 兼容性补丁：解决 torchvision 新版本与旧库的兼容问题
import torch
import torchvision.models
import types

# 创建 torchvision.models.utils 模块（已在新版本中移除）
utils_module = types.ModuleType('torchvision.models.utils')
utils_module.load_state_dict_from_url = torch.hub.load_state_dict_from_url
torchvision.models.utils = utils_module

# 注入到 sys.modules
import sys
sys.modules['torchvision.models.utils'] = utils_module
