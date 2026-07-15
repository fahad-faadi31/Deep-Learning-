import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


df=pd.read_csv("AndGate.csv")

# print(df)

X=df[['x1','x2']]
y=df["target"]
X=torch.tensor(X.values,dtype=torch.float32)
y=torch.tensor(y.values,dtype=torch.float32)
# print(X)
# print(y)

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(2,1)

    def forward(self,x):
        return torch.sigmoid(self.linear(x))

model=Perceptron()
# print(model)

LossFunc=nn.BCELoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)

losses = []

for epoch in range(1000):
    outputs=model(X)
    loss=LossFunc(outputs,y.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if(epoch+1)%10==0:
        print(f"Epoch {epoch+1},Loss:{loss.item():.4f}")

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


with torch.no_grad():
    predictions=model(X)
    predicted=(predictions>=0.5).float()
    print("\nPredictions:")
    print(predicted)
    print("\nActual:")
    print(y.unsqueeze(1))