from Lib_800 import nn
from Lib_800 import loss
from Lib_800 import optim
from Lib_800 import func
import numpy as np

input_size = 3
hidden_size = 10
output_size = 2

class TinyNet(nn.Module):
    def __init__(self):
     super().__init__()

     self.ll1 = nn.Linear(input_size, hidden_size) #(3, 10)
     self.ll2 = nn.Linear(hidden_size, output_size) #->(10, 2)
     
    def forward(self, x):
        x = self.ll1(x)
        x = func.sigmoid(x)
        x = self.ll2(x)
        return x


model = TinyNet()
loss_fn = loss.MSELoss(model)
optimizer = optim.SGD(model, lr=0.01)


X = np.array([[1.0, 2.0, 3.0]])
y = np.array([[10.0, 5.0]]) 

for epcoh in range(1):
    out = model.forward(X)

    loss = loss_fn(out, y)
    optimizer.zero_grad()
    loss_fn.backward()
    optimizer.step()

out = model.forward(X)
all_params = model.parameters()
print("Model Parameters: ", out)