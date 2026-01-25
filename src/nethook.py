'''
Utilities for instrumenting a torch model.
Adapted from: https://github.com/BierOne/ood_coverage/blob/main/openood/postprocessors/nac/nethook.py
'''

import torch
import numpy
import types
from collections import OrderedDict

class InstrumentedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._retained = OrderedDict()
        self._detach_retained = {}
        self._hooked_layer = {}
        self._old_forward = {}
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def retain_layers(self, layernames, detach=True):
        self.add_hooks(layernames)
        for layername in layernames:
            if layername not in self._retained:
                self._retained[layername] = None
                self._detach_retained[layername] = detach

    def retained_layer(self, layername, clear=False):
        result = self._retained[layername]
        if clear:
            self._retained[layername] = None
        return result

    def add_hooks(self, layernames):
        for name, layer in self.model.named_modules():
            if name in layernames:
                if name not in self._hooked_layer:
                    self._hook_layer(layer, name)

    def _hook_layer(self, layer, layername):
        self._hooked_layer[layername] = layername
        self._old_forward[layername] = (layer, layer.__dict__.get('forward', None))
        original_forward = layer.forward
        
        def new_forward(self, *inputs, **kwargs):
            x = original_forward(*inputs, **kwargs)
            return editor._postprocess_forward(x, layername)
            
        editor = self
        layer.forward = types.MethodType(new_forward, layer)

    def _postprocess_forward(self, x, layername):
        if layername in self._retained:
            if self._detach_retained[layername]:
                self._retained[layername] = x.detach()
            else:
                self._retained[layername] = x
        return x

    def close(self):
        for layername in list(self._old_forward.keys()):
            layer, old_forward = self._old_forward[layername]
            if old_forward is None:
                if 'forward' in layer.__dict__:
                    del layer.__dict__['forward']
            else:
                layer.forward = old_forward
            del self._old_forward[layername]
        self._hooked_layer.clear()
