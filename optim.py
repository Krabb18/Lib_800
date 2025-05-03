import numpy as np

class SGD:
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model

    def zero_grad(self):
        for name, module in self.model._submodules.items():
            for param in module.parameters():
                param.grad = 0
        pass

    def step(self):
        for name, module in self.model._submodules.items():
            for param in module.parameters():
                param.value -= self.lr * param.grad