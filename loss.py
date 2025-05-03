import numpy as np

class MSELoss:
    def __init__(self, model):
        self.y_pred = None
        self.y_true = None
        self.model = model

    #wie ein direktewr aufruf
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        error = 2 * (self.y_pred - self.y_true) / self.y_pred.shape[0]
        for name, module in reversed(list(self.model._submodules.items())):
            module.backward(error)
            if hasattr(module, 'W'):
                error = error @ module.W.value.T
