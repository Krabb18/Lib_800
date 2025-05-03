import numpy as np
import math

class Module():
    def __init__(self):
        self._submodules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._submodules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        params = []
        for name, module in self._submodules.items():
            params.extend(module.parameters())
        return params


class Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = 0

class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = Parameter(np.random.randn(input_dim, output_dim))
        self.b = Parameter(np.random.randn(output_dim))
        self.error = None

    def backward(self, error):
    
        # Gradienten für die Gewichtsmatrix W und den Bias b berechnen
        self.W.grad = self.input.T @ error
        self.b.grad = error.sum(axis=0)

    def parameters(self):
        return [self.W, self.b]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input = x
        return x @ self.W.value + self.b.value
    
